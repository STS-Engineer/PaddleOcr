import logging
import time
from pathlib import Path
import contextlib
import os
import urllib.parse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Flask & Networking
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, jsonify, send_file
import requests
import psycopg2

# PDF & Image Processing
import fitz  # PyMuPDF
import cv2
from paddleocr import PaddleOCR, DocImgOrientationClassification

import re
from openai import OpenAI
from psycopg2.extras import Json
from groq import Groq
import json

import uuid
from threading import Thread
from flask import send_from_directory

# --- CONFIGURATION ---
load_dotenv()

UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"
openai_api_key = (
    os.getenv("OPENAI_API_KEY")
)
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

TEMP_PDF_FOLDER = Path('/tmp/temp_pdfs')  # TTL 30 min
TEMP_PDF_FOLDER.mkdir(exist_ok=True, parents=True)

# DB Config
DB_CONFIG = {
    "host": "avo-adb-002.postgres.database.azure.com",
    "database": "Costing_DB",
    "user": "administrationSTS",
    "password": "St$@0987"
}

# GitHub Config
GITHUB_OWNER = "STS-Engineer"
GITHUB_REPO = "RFQ-back"
GITHUB_BRANCH = "main"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# ----------------- MODEL INITIALIZATION ----------------- #

paddle_engine = None
doc_ori = None

def init_models():
    """Initialize PaddleOCR and Orientation models."""
    global paddle_engine, doc_ori
    
    try:
        logging.info("Loading PaddleOCR (lang=ch)...")
        # Main OCR Engine
        paddle_engine = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=True, 
            use_doc_unwarping=False,            
            use_textline_orientation=True     
        )
        
        # Orientation Classification Model
        doc_ori = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        
        logging.info("Models initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL: Failed to load models: {e}")

# Pre-load models
init_models()

# ----------------- HELPER FUNCTIONS ----------------- #

def cleanup_old_files(folder, max_age_hours=24):
    """Deletes files older than max_age_hours to prevent disk clutter."""
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        for file_path in folder.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    os.remove(file_path)
    except Exception:
        pass # Silent fail is fine for cleanup


def cleanup_temp_pdfs(interval_seconds=300, max_age_seconds=30 * 60):
    """Delete temp pdfs older than 30 minutes."""
    while True:
        now = time.time()
        try:
            for f in TEMP_PDF_FOLDER.iterdir():
                if not f.is_file():
                    continue
                age = now - f.stat().st_mtime
                if age > max_age_seconds:
                    with contextlib.suppress(Exception):
                        f.unlink()
        except Exception:
            pass
        time.sleep(interval_seconds)

Thread(target=cleanup_temp_pdfs, daemon=True).start()


def extract_material_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    s = re.sub(r"[_\-]+", " ", stem).strip()

    s = re.sub(r"(?i)^\s*data\s*sheet\s*", "", s)
    s = re.sub(r"(?i)^\s*datasheet\s*", "", s)
    s = re.sub(r"(?i)^\s*technical\s*data\s*sheet\s*", "", s)
    s = re.sub(r"(?i)^\s*fiche\s*technique\s*", "", s)
    s = re.sub(r"(?i)^\s*fiche\s*de\s*donn[eé]es\s*", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s or "UNKNOWN"


def extract_reference_from_msds_filename(filename: str) -> str:
    """
    Extracts reference number from MSDS filenames like:
    '6600125 - TIMCAL - SDS - English.pdf' -> '6600125'
    Returns None if pattern doesn't match.
    """
    stem = Path(filename).stem
    # Extract first part before ' - '
    parts = stem.split(' - ')
    if parts and parts[0].strip():
        ref = parts[0].strip()
        # Verify it looks like a reference (digits)
        if ref.replace(' ', '').isdigit():
            return ref
    return None


def extract_reference_from_inspection_filename(filename: str) -> str:
    """
    Extracts reference number from Inspection/Control sheet filenames like:
    '6600125.xls' -> '6600125'
    '6600125.xlsx' -> '6600125'
    Returns None if pattern doesn't match.
    """
    stem = Path(filename).stem
    # For inspection sheets, the stem IS the reference
    ref = stem.strip()
    # Verify it looks like a reference (digits)
    if ref.replace(' ', '').isdigit():
        return ref
    return None


# ============== GROQ LLM INTEGRATION ============== #

def build_specifications_prompt(ocr_text: str, material_name: str) -> str:
    """
    Build a prompt for Groq LLM to extract and structure specifications from OCR text.
    """
    prompt = f"""You are an expert materials engineer assistant. Your task is to extract material specifications from OCR-extracted text and structure them into a JSON object.

## Material Information
- Material Name: {material_name}
- OCR Text: {ocr_text}

## Your Task

Extract all relevant specifications from the OCR text and structure them into a JSON object with the following schema:

{{
  "document_info": {{
    "document_type": "string (e.g., 'datasheet', 'technical_sheet')",
    "supplier": "string or null",
    "date": "string or null"
  }},
  "material_identification": {{
    "material_name": "{material_name}",
    "chemical_name": "string or null",
    "cas_number": "string or null",
    "grade": "string or null"
  }},
  "composition": [
    {{
      "property": "component name",
      "value": "percentage or concentration",
      "unit": "% or ppm",
      "condition": null,
      "source_page": 1,
      "confidence": "high|medium|low"
    }}
  ],
  "physical_properties": [
    {{
      "property": "density|color|appearance|particle_size|etc",
      "value": "measured value",
      "unit": "g/cm³|μm|etc",
      "condition": "at 20°C or null",
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "mechanical_properties": [
    {{
      "property": "tensile_strength|hardness|modulus|etc",
      "value": "numeric value",
      "unit": "MPa|GPa|etc",
      "condition": "test conditions or null",
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "thermal_properties": [
    {{
      "property": "melting_point|thermal_conductivity|expansion|etc",
      "value": "numeric value",
      "unit": "°C|W/m·K|etc",
      "condition": "temperature range or null",
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "electrical_properties": [
    {{
      "property": "conductivity|resistivity|dielectric_strength|etc",
      "value": "numeric value",
      "unit": "S/m|Ω·m|kV/mm|etc",
      "condition": "frequency or temperature or null",
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "chemical_resistance": [
    {{
      "property": "resistance_to_acid|alcohol|solvent|etc",
      "value": "resistant|not_resistant|limited_resistance",
      "unit": null,
      "condition": "chemical name",
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "standards_and_certifications": [
    {{
      "property": "standard_name",
      "value": "certification_status",
      "unit": null,
      "condition": null,
      "source_page": page_number,
      "confidence": "high|medium|low"
    }}
  ],
  "processing_and_notes": {{
    "processing_temperature": "string or null",
    "special_handling": "string or null",
    "storage_conditions": "string or null",
    "notes": "additional important notes or null"
  }},
  "raw_excerpts_by_page": {{
    "1": "exact OCR text from page 1",
    "2": "exact OCR text from page 2"
  }}
}}

## Instructions

1. **Extract accurately**: Only include properties that are explicitly mentioned in the OCR text.
2. **Confidence levels**: 
   - "high": Value is clearly stated with units
   - "medium": Value requires minor interpretation or unit conversion
   - "low": Value is uncertain, implied, or partially visible
3. **Units**: Always standardize units to SI. If original unit differs, note in the value.
4. **Page tracking**: Record which page each property was found on.
5. **Missing values**: Leave value blank ("") if not found, set confidence to "low", never fabricate data.
6. **Do not add extra fields** outside the schema above.

## Output

Return ONLY valid JSON object. Do not include markdown, code blocks, or explanations. Start with {{ and end with }}.
"""
    return prompt


def process_ocr_with_groq(ocr_text: str, material_name: str) -> dict:
    """
    Use Groq LLM to extract and structure specifications from OCR text.
    Returns the parsed SpecificationsJson or an error dict.
    """
    if not groq_client:
        return {
            "success": False,
            "error": "Groq API key not configured"
        }

    try:
        prompt = build_specifications_prompt(ocr_text, material_name)
        
        logging.info("Sending OCR text to Groq LLM for specifications extraction...")
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = message.choices[0].message.content.strip()
        
        # Parse JSON response
        specs_json = json.loads(response_text)
        
        logging.info("Successfully extracted specifications via Groq LLM")
        return {
            "success": True,
            "specifications": specs_json
        }

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Groq response: {e}")
        return {
            "success": False,
            "error": f"Invalid JSON response from LLM: {e}"
        }
    except Exception as e:
        logging.error(f"Error processing with Groq: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_file(filename):
    try:
        return send_from_directory(str(TEMP_PDF_FOLDER), filename)
    except Exception:
        return jsonify({"success": False, "error": "temp_file_not_found"}), 404


@app.route("/upload-temp-pdf", methods=["POST"])
def upload_temp_pdf():
    """
    JSON accepté (ordre de priorité):
      1) openaiFileIdRefs: [ {id, download_link?, name?, mime_type?}, ... ]
      2) download_link: Azure SAS URL (preferred)
      3) openai_file_id: fallback OpenAI Files API

    Returns:
      - temp_filename + temp_url (valid until cleanup ~30 min)
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}

    # --- NEW: accept openaiFileIdRefs ---
    refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    original_name = data.get("filename") or "uploaded.pdf"

    # If openaiFileIdRefs is provided, it has priority
    if refs and isinstance(refs, list) and len(refs) > 0:
        first_ref = refs[0] if isinstance(refs[0], dict) else {"id": str(refs[0])}

        # Prefer download_link if present
        dl = first_ref.get("download_link")
        if dl:
            download_link = dl

        # Keep file id as fallback
        fid = first_ref.get("id")
        if fid and not openai_file_id:
            openai_file_id = fid

        # Try to use original filename
        ref_name = first_ref.get("name")
        if ref_name:
            original_name = ref_name

    if not download_link and not openai_file_id:
        return jsonify({
            "success": False,
            "error": "Provide 'openaiFileIdRefs', 'download_link' (Azure SAS) or 'openai_file_id'"
        }), 400

    try:
        pdf_bytes = None

        # 1) Preferred: download from link (Azure SAS OR Actions temp link)
        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Failed to download from link (status={r.status_code}). Maybe expired?"
                }), 400
            pdf_bytes = r.content

        # 2) Fallback: OpenAI Files API
        if pdf_bytes is None and openai_file_id:
            file_metadata = client.files.retrieve(openai_file_id)
            if getattr(file_metadata, "filename", None):
                original_name = file_metadata.filename
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        # Save temp file
        safe = secure_filename(original_name) or "uploaded.pdf"
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"

        temp_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{safe}"
        temp_path = TEMP_PDF_FOLDER / temp_filename

        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        temp_url = f"{request.host_url.rstrip('/')}/temp_files/{temp_filename}"

        return jsonify({
            "success": True,
            "temp_filename": temp_filename,
            "temp_url": temp_url,
            "expires_in_seconds": 30 * 60
        }), 200

    except Exception as e:
        logging.error(f"Temp upload failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload-temp-pdf-and-ocr", methods=["POST"])
def upload_temp_pdf_and_ocr():
    """
    Combined endpoint:
      - Upload/download PDF (same inputs as /upload-temp-pdf)
      - Save as temp file
      - Run OCR (same behavior as /process-pdf-to-ocr)
      - Return temp_url + OCR results
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required. Provide 'openaiFileIdRefs', 'download_link' OR 'openai_file_id'"
        }), 400

    data = request.get_json(silent=True) or {}

    refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    max_pages = int(data.get("max_pages", 20))
    original_name = data.get("filename") or "uploaded.pdf"

    if refs and isinstance(refs, list) and len(refs) > 0:
        first_ref = refs[0] if isinstance(refs[0], dict) else {"id": str(refs[0])}
        dl = first_ref.get("download_link")
        if dl:
            download_link = dl
        fid = first_ref.get("id")
        if fid and not openai_file_id:
            openai_file_id = fid
        ref_name = first_ref.get("name")
        if ref_name:
            original_name = ref_name

    if not download_link and not openai_file_id:
        return jsonify({
            "success": False,
            "error": "Provide 'openaiFileIdRefs', 'download_link' (Azure SAS) or 'openai_file_id'"
        }), 400

    timestamp = int(time.time())
    temp_path = None

    try:
        pdf_bytes = None

        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Failed to download from link (status={r.status_code}). Maybe expired?"
                }), 400
            pdf_bytes = r.content

        if pdf_bytes is None and openai_file_id:
            file_metadata = client.files.retrieve(openai_file_id)
            if getattr(file_metadata, "filename", None):
                original_name = file_metadata.filename
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        safe = secure_filename(original_name) or "uploaded.pdf"
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"

        temp_filename = f"{uuid.uuid4().hex}_{timestamp}_{safe}"
        temp_path = TEMP_PDF_FOLDER / temp_filename
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        temp_url = f"{request.host_url.rstrip('/')}/temp_files/{temp_filename}"

        material_name = extract_material_name_from_filename(safe)

        cleanup_old_files(OUTPUT_FOLDER)
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        mat = fitz.Matrix(2.0, 2.0)

        processed_pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)

            raw_filename = f"oa_raw_{i+1}_{timestamp}.png"
            raw_path = OUTPUT_FOLDER / raw_filename
            pix.save(str(raw_path))

            upright_filename = f"oa_upright_{i+1}_{timestamp}.png"
            upright_path = OUTPUT_FOLDER / upright_filename
            angle = save_upright_image(raw_path, upright_path)
            raw_path.unlink(missing_ok=True)

            ocr_text_list = run_paddle_ocr_on_file(upright_path)

            processed_pages.append({
                "page": i + 1,
                "rotation_angle": angle,
                "ocr_text": ocr_text_list
            })

        doc.close()

        reference = None
        if original_name and ("SDS" in original_name.upper() or "MSDS" in original_name.upper()):
            reference = extract_reference_from_msds_filename(original_name)

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "material_name": material_name,
            "reference": reference,
            "total_pages": total_pages,
            "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "pages": processed_pages,
            "temp_filename": temp_filename,
            "temp_url": temp_url,
            "expires_in_seconds": 30 * 60
        }), 200

    except Exception as e:
        logging.error(f"Upload+OCR failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def save_upright_image(in_path: Path, out_path: Path) -> int:
    """
    Predicts orientation, rotates the image using OpenCV, and saves it.
    Returns the rotation angle applied (0, 90, 180, 270).
    """
    if doc_ori is None:
        return -1

    out = doc_ori.predict(str(in_path), batch_size=1)
    if not out:
        return -1

    res = out[0]
    d = res.to_dict() if hasattr(res, "to_dict") else {}
    angle = int((d.get("res", {}).get("label_names") or ["0"])[0])

    img = cv2.imread(str(in_path))
    if img is None:
        return -1

    # Rotate manually based on prediction
    if angle == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imwrite(str(out_path), img)
    return angle

def run_paddle_ocr_on_file(img_path: Path):
    """Runs PaddleOCR on a specific image file."""
    if paddle_engine is None:
        logging.error("PaddleOCR engine not initialized.")
        return []

    try:
        result = paddle_engine.predict(
            str(img_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_det_limit_side_len=2048,
            text_det_limit_type="max",
            text_det_thresh=0.2,
            text_det_box_thresh=0.2,
            text_det_unclip_ratio=2.0,
            text_rec_score_thresh=0.0,
            return_word_box=True,
        )

        if not result:
            return []

        detected_texts = []
        for res in result:
            d = res.to_dict() if hasattr(res, "to_dict") else res
            if not isinstance(d, dict):
                continue
            
            payload = d.get("res", d)
            texts = payload.get("rec_texts") or []
            
            for t in texts:
                if t:
                    detected_texts.append(str(t))

        return detected_texts

    except Exception:
        logging.exception(f"PaddleOCR failed on {img_path}")
        return []

# ----------------- ENDPOINTS ----------------- #

@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    """Required to view the images returned by the process route."""
    try:
        return send_file(str(OUTPUT_FOLDER / secure_filename(filename)), mimetype='image/jpeg')
    except Exception:
        return jsonify({"success": False, "error": "Image not found"}), 404

@app.route('/download-image/<filename>', methods=['GET'])
def download_image(filename):
    """Required to download the images returned by the process route."""
    try:
        return send_file(str(OUTPUT_FOLDER / secure_filename(filename)), as_attachment=True)
    except Exception:
        return jsonify({"success": False, "error": "Image not found"}), 404

@app.route('/process-rfq-id-to-images', methods=['POST'])
def process_rfq_id_to_images():
    """
    Main Route:
    1. Fetches PDF path from DB using RFQ_ID.
    2. Downloads PDF from GitHub.
    3. Converts to Images -> Rotates Upright -> Runs OCR.
    4. Returns JSON with image URLs and Extracted Text.
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')
    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    mat = fitz.Matrix(2.0, 2.0) # High-res zoom
    local_pdf_path = None
    conn = None
    download_url_page_1 = None

    try:
        # 1. DB Fetch & Search for Drawing
        logging.info(f"Fetching RFQ and searching for drawing: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # We query the file path column. We use ILIKE to find rows where 'drawing' 
        # exists anywhere in that path string.
        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()
        
        if not result or not result[0]:
            return jsonify({"success": False, "error": "RFQ ID or File Path not found"}), 404
        
        # result[0] looks like: "{/path/file1.pdf,/path/Drawing_abc.pdf}"
        raw_paths = result[0].strip("{}").split(",")
        
        # Logic to prioritize the file containing 'drawing'
        rfq_path_db = None
        for path in raw_paths:
            if 'drawing' in path.lower():
                rfq_path_db = path.strip()
                break
        
        # Fallback to the first file if no 'drawing' is found
        if not rfq_path_db and raw_paths:
            rfq_path_db = raw_paths[0].strip()
        
        logging.info(f"Target file identified: {rfq_path_db}")
        
        # 2. GitHub Download
        clean_path = rfq_path_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"
        
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub error: {resp.status_code}"}), 400
            
        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name
        
        with open(local_pdf_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # 3. Process Pages
        cleanup_old_files(OUTPUT_FOLDER)
        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        max_pages = int(data.get('max_pages', 20))
        
        processed_pages = []
        timestamp = int(time.time())
        base_url = request.host_url.rstrip('/')

        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)

            # Save initial raw render
            raw_filename = f"{rfq_id}_page_{i+1}_{timestamp}.png"
            raw_path = OUTPUT_FOLDER / raw_filename
            pix.save(str(raw_path))

            # Correct Orientation (Save as new file)
            upright_filename = f"{rfq_id}_page_{i+1}_{timestamp}_upright.png"
            upright_path = OUTPUT_FOLDER / upright_filename
            angle = save_upright_image(raw_path, upright_path)

            # Cleanup raw immediately
            raw_path.unlink(missing_ok=True)

            # OCR on the upright image
            logging.info(f"Running OCR on page {i+1} (angle={angle})...")
            ocr_text_list = run_paddle_ocr_on_file(upright_path)

            # Prepare Response URLs
            view_url = f"{base_url}/images/{upright_filename}"
            dl_url   = f"{base_url}/download-image/{upright_filename}"

            processed_pages.append({
                "page": i + 1,
                "url": view_url,
                "lien_pour_telecharger_l_image": dl_url,
                "filename": upright_filename,
                "rotation_angle": angle,
                "ocr_text": ocr_text_list
            })

            if i == 0:
                download_url_page_1 = dl_url

        doc.close()
        
        if not download_url_page_1 and processed_pages:
            download_url_page_1 = processed_pages[0]["lien_pour_telecharger_l_image"]

        return jsonify({
            "success": True,
            "rfq_id": rfq_id,
            "source_path": rfq_path_db,
            "total_pages": total_pages,
            "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "download_url_page_1_png": download_url_page_1,
            "images": processed_pages
        }), 200

    except Exception as e:
        logging.error(f"RFQ Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if conn:
            with contextlib.suppress(Exception):
                cur.close()
                conn.close()
        if local_pdf_path and local_pdf_path.exists():
            with contextlib.suppress(Exception):
                os.remove(local_pdf_path)
@app.route("/process-pdf-to-ocr", methods=["POST"])
def process_pdf_to_ocr():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required. Provide 'openaiFileIdRefs', 'download_link' OR 'openai_file_id'"
        }), 400

    data = request.get_json()
    
    # Handle openaiFileIdRefs array (Claude's format)
    openai_file_id_refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    max_pages = int(data.get("max_pages", 20))
    original_filename = None

    # Extract from openaiFileIdRefs if present (priority #1)
    if openai_file_id_refs and isinstance(openai_file_id_refs, list) and len(openai_file_id_refs) > 0:
        first_ref = openai_file_id_refs[0]
        download_link = first_ref.get("download_link")
        original_filename = first_ref.get("name", "uploaded.pdf")
        logging.info(f"Using openaiFileIdRefs: {original_filename}")

    if not download_link and not openai_file_id:
        return jsonify({
            "success": False,
            "error": "Missing 'openaiFileIdRefs', 'download_link' or 'openai_file_id'"
        }), 400

    timestamp = int(time.time())
    local_pdf_path = None

    try:
        # -----------------------------
        # 1) Get PDF bytes
        # -----------------------------
        pdf_bytes = None

        # A) preferred: direct download link (temp 30 min)
        if download_link:
            logging.info(f"Downloading from link: {download_link[:100]}...")
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Download link failed (status={r.status_code}). Maybe expired?"
                }), 400
            pdf_bytes = r.content

            # Use original_filename if already set from openaiFileIdRefs
            if not original_filename:
                # Try to infer filename from URL path
                try:
                    p = urllib.parse.urlparse(download_link).path
                    original_filename = Path(p).name or "uploaded.pdf"
                except Exception:
                    original_filename = "uploaded.pdf"

        # B) fallback: OpenAI Files API
        if pdf_bytes is None and openai_file_id:
            logging.info(f"Using OpenAI file ID: {openai_file_id}")
            file_metadata = client.files.retrieve(openai_file_id)
            original_filename = getattr(file_metadata, "filename", None) or "uploaded.pdf"
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        # -----------------------------
        # 2) Save locally (temp 30 min)
        # -----------------------------
        safe_name = secure_filename(original_filename or "uploaded.pdf") or "uploaded.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        material_name = extract_material_name_from_filename(safe_name)

        unique_pdf_name = f"{uuid.uuid4().hex}_{timestamp}_{safe_name}"
        local_pdf_path = TEMP_PDF_FOLDER / unique_pdf_name

        with open(local_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        logging.info(f"PDF saved to: {local_pdf_path}")

        # -----------------------------
        # 3) OCR
        # -----------------------------
        cleanup_old_files(OUTPUT_FOLDER)
        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        mat = fitz.Matrix(2.0, 2.0)

        processed_pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)

            raw_filename = f"oa_raw_{i+1}_{timestamp}.png"
            raw_path = OUTPUT_FOLDER / raw_filename
            pix.save(str(raw_path))

            upright_filename = f"oa_upright_{i+1}_{timestamp}.png"
            upright_path = OUTPUT_FOLDER / upright_filename
            angle = save_upright_image(raw_path, upright_path)
            raw_path.unlink(missing_ok=True)

            ocr_text_list = run_paddle_ocr_on_file(upright_path)

            processed_pages.append({
                "page": i + 1,
                "rotation_angle": angle,
                "ocr_text": ocr_text_list
            })

        doc.close()

        # Detect if this is an MSDS file and extract reference
        reference = None
        if original_filename and ('SDS' in original_filename.upper() or 'MSDS' in original_filename.upper()):
            reference = extract_reference_from_msds_filename(original_filename)
            logging.info(f"Detected MSDS file. Extracted reference: {reference}")

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "material_name": material_name,
            "reference": reference,  # NEW: Include extracted reference for MSDS
            "total_pages": total_pages,
            "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "pages": processed_pages
        }), 200

    except Exception as e:
        logging.error(f"PDF Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Temp PDF will be cleaned by background thread after 30 min
        pass





@app.route("/save-ocr-result", methods=["POST"])
def save_ocr_result():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required"
        }), 400

    data = request.get_json(silent=True) or {}

    # --- Required fields ---
    material_name = data.get("material_name")
    specifications = data.get("specifications")  # 🔥 AI structured JSON
    source = data.get("source_type", "datasheet")
    reference = data.get("reference")
    type_matiere = data.get("type_matiere") or material_name

    # --- Validation ---
    if not material_name:
        return jsonify({
            "success": False,
            "error": "Missing material_name"
        }), 400

    if not specifications:
        return jsonify({
            "success": False,
            "error": "Missing specifications JSON"
        }), 400

    try:
        matiere_id, fiche_id = save_material_and_fiche(
            material_name=material_name,
            type_matiere=type_matiere,
            specs_json=specifications,   # ✅ Direct AI JSON
            source=source,
            reference=reference
        )

        return jsonify({
            "success": True,
            "matiere_id": matiere_id,
            "fiche_id": fiche_id,
            "source": source,
            "reference": reference
        }), 200

    except ValueError as ve:
        # Handle specific business logic errors (e.g., material not found)
        logging.error(f"Validation error: {ve}")
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        logging.error(f"Save OCR Result Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/specifications-by-reference", methods=["GET"])
def get_specifications_by_reference():
    reference = request.args.get("reference")
    if not reference:
        return jsonify({
            "success": False,
            "error": "Missing reference"
        }), 400

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.spec_id, s.fiche_id, s.source_type, s.donnees
                    FROM public.matieres m
                    JOIN public.fiches_matieres f ON f.matiere_id = m.matiere_id
                    JOIN public.specifications s ON s.fiche_id = f.fiche_id
                    WHERE m.reference = %s
                    ORDER BY s.spec_id ASC
                    """,
                    (reference,)
                )
                rows = cur.fetchall()

        specs = [
            {
                "spec_id": r[0],
                "fiche_id": r[1],
                "source_type": r[2],
                "donnees": r[3]
            }
            for r in rows
        ]

        return jsonify({
            "success": True,
            "reference": reference,
            "specifications": specs
        }), 200

    except Exception as e:
        logging.error(f"Get specifications by reference error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/update-specification-by-reference", methods=["PUT"])
def update_specification_by_reference():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required"
        }), 400

    data = request.get_json(silent=True) or {}
    reference = data.get("reference")
    spec_id = data.get("spec_id")
    donnees = data.get("donnees")

    if not spec_id:
        return jsonify({"success": False, "error": "Missing spec_id"}), 400
    if donnees is None:
        return jsonify({"success": False, "error": "Missing donnees"}), 400

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                if reference:
                    cur.execute(
                        """
                        SELECT m.matiere_id, f.fiche_id, s.spec_id
                        FROM public.matieres m
                        JOIN public.fiches_matieres f ON f.matiere_id = m.matiere_id
                        JOIN public.specifications s ON s.fiche_id = f.fiche_id
                        WHERE m.reference = %s AND s.spec_id = %s
                        LIMIT 1
                        """,
                        (reference, spec_id)
                    )
                else:
                    cur.execute(
                        """
                        SELECT f.matiere_id, s.fiche_id, s.spec_id
                        FROM public.specifications s
                        JOIN public.fiches_matieres f ON f.fiche_id = s.fiche_id
                        WHERE s.spec_id = %s
                        LIMIT 1
                        """,
                        (spec_id,)
                    )
                row = cur.fetchone()
                if not row:
                    return jsonify({
                        "success": False,
                        "error": "Specification not found for spec_id"
                    }), 404

                matiere_id, fiche_id, spec_id = row
                cur.execute(
                    """
                    UPDATE public.specifications
                    SET donnees = %s,
                        derniere_modification = NOW()
                    WHERE spec_id = %s
                    """,
                    (Json(donnees), spec_id)
                )

        return jsonify({
            "success": True,
            "matiere_id": matiere_id,
            "fiche_id": fiche_id,
            "spec_id": spec_id,
            "reference": reference
        }), 200

    except Exception as e:
        logging.error(f"Update specification by reference error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


def save_material_and_fiche(material_name: str, type_matiere: str, specs_json: dict, source: str, reference: str = None):
    """
    Sauvegarde la matière, crée la fiche technique, 
    puis insère les données dans la table specifications.
    
    If reference is provided (MSDS case), searches for existing material by reference.
    Otherwise, searches by material_name (datasheet case).
    """
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                matiere_id = None
                
                # NEW: Priority search by reference if provided (for MSDS)
                if reference:
                    logging.info(f"Searching for material by reference: {reference}")
                    cur.execute(
                        "SELECT matiere_id, nom_matiere FROM public.matieres WHERE reference = %s", 
                        (reference,)
                    )
                    row = cur.fetchone()
                    if row:
                        matiere_id = row[0]
                        existing_name = row[1]
                        logging.info(f"Found existing material '{existing_name}' (ID: {matiere_id}) with reference {reference}")
                        # ✅ Material exists - DO NOT update, just use existing matiere_id
                    else:
                        # For MSDS, material MUST exist
                        raise ValueError(
                            f"Material with reference '{reference}' not found in database. "
                            f"MSDS can only be added to existing materials. Please ensure the material "
                            f"datasheet has been processed first."
                        )
                
                # Fallback: Search by material_name (datasheet behavior - only if reference not found)
                if not matiere_id:
                    logging.info(f"Searching for material by name: {material_name}")
                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE nom_matiere = %s",
                        (material_name,)
                    )
                    row = cur.fetchone()

                    if row:
                        matiere_id = row[0]
                        logging.info(f"Found existing material by name (ID: {matiere_id})")
                        # Only update if we have new information to add (reference or type)
                        if reference or type_matiere:
                            cur.execute(
                                """UPDATE public.matieres 
                                   SET type_matiere = COALESCE(%s, type_matiere),
                                       reference = COALESCE(%s, reference),
                                       date_mise_a_jour = NOW()
                                   WHERE matiere_id = %s""",
                                (type_matiere, reference, matiere_id)
                            )
                            logging.info(f"Updated material {matiere_id} with new info")
                    else:
                        # Create NEW material (datasheet case only)
                        logging.info(f"Creating new material: {material_name}")
                        cur.execute(
                            """INSERT INTO public.matieres 
                               (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
                               VALUES (%s, %s, %s, NOW(), NOW())
                               RETURNING matiere_id""",
                            (material_name, type_matiere, reference)
                        )
                        matiere_id = cur.fetchone()[0]
                        logging.info(f"Created new material with ID: {matiere_id}")

                # 2. Check if fiche already exists for this material + source_type
                logging.info(f"Checking for existing fiche for material {matiere_id} with source_type='{source}'")
                cur.execute(
                    """SELECT f.fiche_id, s.spec_id 
                       FROM public.fiches_matieres f
                       JOIN public.specifications s ON s.fiche_id = f.fiche_id
                       WHERE f.matiere_id = %s AND s.source_type = %s
                       ORDER BY f.fiche_id DESC
                       LIMIT 1""",
                    (matiere_id, source)
                )
                existing = cur.fetchone()
                
                if existing:
                    # UPDATE existing specification
                    fiche_id, spec_id = existing
                    logging.info(f"Found existing fiche {fiche_id} and spec {spec_id} - UPDATING donnees")
                    cur.execute(
                        """UPDATE public.specifications 
                           SET donnees = %s,
                               derniere_modification = NOW()
                           WHERE spec_id = %s""",
                        (Json(specs_json), spec_id)
                    )
                    logging.info(f"✅ Updated specifications for fiche {fiche_id}")
                else:
                    # CREATE new fiche + specification
                    logging.info(f"No existing fiche found - creating new fiche for material {matiere_id}")
                    cur.execute(
                        """INSERT INTO public.fiches_matieres 
                           (matiere_id, date_creation_fiche, derniere_modification)
                           VALUES (%s, NOW(), NOW())
                           RETURNING fiche_id""",
                        (matiere_id,)
                    )
                    fiche_id = cur.fetchone()[0]
                    logging.info(f"Created fiche with ID: {fiche_id}")

                    # Insert new specifications
                    cur.execute(
                        """INSERT INTO public.specifications 
                           (fiche_id, source_type, donnees, date_creation, derniere_modification)
                           VALUES (%s, %s, %s, NOW(), NOW())""",
                        (fiche_id, source, Json(specs_json))
                    )
                    logging.info(f"✅ Inserted new specifications for fiche {fiche_id}")

        return matiere_id, fiche_id
    finally:
        conn.close()


# ============== GROQ-INTEGRATED ENDPOINT ============== #

@app.route("/process-pdf-with-groq", methods=["POST"])
def process_pdf_with_groq():
    """
    Comprehensive endpoint to:
    1. Upload/download PDF
    2. Run OCR to extract text
    3. Use Groq LLM to construct SpecificationsJson
    4. Save results to database
    
    Request JSON:
    {
        "openaiFileIdRefs": [...],  // OR
        "download_link": "url",      // OR  
        "openai_file_id": "id",
        "filename": "optional.pdf",
        "material_name": "optional - uses filename if not provided",
        "type_matiere": "optional",
        "source_type": "datasheet|msds|certificate|other (default: datasheet)",
        "reference": "optional - required for MSDS",
        "max_pages": 20
    }
    
    Response:
    {
        "success": true,
        "matiere_id": 123,
        "fiche_id": 456,
        "reference": "ref123",
        "source": "datasheet",
        "ocr_pages": 5,
        "specifications": {...}
    }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required"
        }), 400

    data = request.get_json(silent=True) or {}
    
    # --- Parse request parameters ---
    refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    original_name = data.get("filename") or "uploaded.pdf"
    material_name = data.get("material_name")
    type_matiere = data.get("type_matiere")
    source_type = data.get("source_type", "datasheet")
    reference = data.get("reference")
    max_pages = data.get("max_pages", 20)

    # --- Validate source_type ---
    valid_sources = ["datasheet", "msds", "certificate", "feuille_de_controle_excel_file", "other"]
    if source_type not in valid_sources:
        return jsonify({
            "success": False,
            "error": f"Invalid source_type. Must be one of: {', '.join(valid_sources)}"
        }), 400

    # --- Handle openaiFileIdRefs ---
    if refs and isinstance(refs, list) and len(refs) > 0:
        first_ref = refs[0] if isinstance(refs[0], dict) else {"id": str(refs[0])}
        dl = first_ref.get("download_link")
        if dl:
            download_link = dl
        fid = first_ref.get("id")
        if fid and not openai_file_id:
            openai_file_id = fid
        ref_name = first_ref.get("name")
        if ref_name:
            original_name = ref_name

    if not download_link and not openai_file_id:
        return jsonify({
            "success": False,
            "error": "Provide 'openaiFileIdRefs', 'download_link', or 'openai_file_id'"
        }), 400

    temp_path = None
    try:
        # --- Download PDF ---
        pdf_bytes = None
        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Failed to download (status={r.status_code})"
                }), 400
            pdf_bytes = r.content

        if pdf_bytes is None and openai_file_id:
            file_metadata = client.files.retrieve(openai_file_id)
            if getattr(file_metadata, "filename", None):
                original_name = file_metadata.filename
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        # --- Save temp PDF ---
        safe = secure_filename(original_name) or "uploaded.pdf"
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"

        temp_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{safe}"
        temp_path = TEMP_PDF_FOLDER / temp_filename

        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        # --- Extract material name if not provided ---
        if not material_name:
            material_name = extract_material_name_from_filename(safe)

        # --- Run OCR ---
        logging.info(f"Running OCR on {safe}...")
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages)
        
        ocr_text_by_page = {}
        all_ocr_text = ""
        
        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                
                # Save to temp file for OCR
                timestamp_unique = int(time.time() * 1000) + page_num  # Unique per page
                temp_img_path = OUTPUT_FOLDER / f"temp_ocr_json_{page_num}_{timestamp_unique}.png"
                pix.save(str(temp_img_path))
                
                # Use existing OCR function that works
                ocr_text_list = run_paddle_ocr_on_file(temp_img_path)
                page_text = "\n".join(ocr_text_list)
                
                ocr_text_by_page[str(page_num + 1)] = page_text
                all_ocr_text += f"\n=== PAGE {page_num + 1} ===\n{page_text}"
                
                # Cleanup temp image
                temp_img_path.unlink(missing_ok=True)
                
            except Exception as e:
                logging.warning(f"OCR failed for page {page_num + 1}: {e}")
                ocr_text_by_page[str(page_num + 1)] = f"[OCR ERROR: {e}]"

        doc.close()
        logging.info(f"OCR completed for {pages_to_process} pages")

        # --- Use Groq to construct SpecificationsJson ---
        groq_result = process_ocr_with_groq(all_ocr_text, material_name)
        if not groq_result.get("success"):
            return jsonify({
                "success": False,
                "error": f"Groq processing failed: {groq_result.get('error', 'Unknown error')}"
            }), 500

        specs_json = groq_result.get("specifications", {})
        
        # Add raw excerpts
        if "raw_excerpts_by_page" not in specs_json:
            specs_json["raw_excerpts_by_page"] = {}
        specs_json["raw_excerpts_by_page"].update(ocr_text_by_page)

        # --- Validate SpecificationsJson ---
        if not isinstance(specs_json, dict):
            return jsonify({
                "success": False,
                "error": "Invalid specifications object from LLM"
            }), 500

        # --- Save to database ---
        matiere_id, fiche_id = save_material_and_fiche(
            material_name=material_name,
            type_matiere=type_matiere or material_name,
            specs_json=specs_json,
            source=source_type,
            reference=reference
        )

        logging.info(f"✅ Successfully processed PDF: matiere_id={matiere_id}, fiche_id={fiche_id}")

        return jsonify({
            "success": True,
            "matiere_id": matiere_id,
            "fiche_id": fiche_id,
            "reference": reference,
            "source": source_type,
            "material_name": material_name,
            "ocr_pages": pages_to_process,
            "total_pages": total_pages,
            "specifications": specs_json
        }), 200

    except Exception as e:
        logging.error(f"Process PDF with Groq error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ============== FORM-DATA ENDPOINT FOR FILE UPLOADS ============== #

@app.route("/process-pdf-with-groq-form", methods=["POST"])
def process_pdf_with_groq_form():
    """
    Form-data version of /process-pdf-with-groq for direct file uploads.
    Ideal for testing in Apidog or Postman with multipart/form-data.
    
    Form Fields:
    - pdf_file (file): The PDF file to process (required)
    - material_name (text): Name of material (optional, extracted from filename if omitted)
    - type_matiere (text): Material type/category (optional)
    - source_type (text): datasheet|msds|certificate|other (default: datasheet)
    - reference (text): Material reference code (required for MSDS)
    - max_pages (number): Max pages to process (default: 20)
    
    Returns: JSON with matiere_id, fiche_id, specifications
    """
    try:
        # --- Parse form fields ---
        material_name = request.form.get("material_name")
        type_matiere = request.form.get("type_matiere")
        source_type = request.form.get("source_type", "datasheet")
        reference = request.form.get("reference")
        max_pages = request.form.get("max_pages", 20, type=int)
        
        # --- Validate source_type ---
        valid_sources = ["datasheet", "msds", "certificate", "feuille_de_controle_excel_file", "other"]
        if source_type not in valid_sources:
            return jsonify({
                "success": False,
                "error": f"Invalid source_type. Must be one of: {', '.join(valid_sources)}"
            }), 400
        
        # --- Check for uploaded file ---
        if "pdf_file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded. Provide 'pdf_file' in form-data"
            }), 400
        
        file = request.files["pdf_file"]
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({
                "success": False,
                "error": "Only PDF files are supported"
            }), 400
        
        # --- Read file content ---
        pdf_bytes = file.read()
        if not pdf_bytes:
            return jsonify({
                "success": False,
                "error": "File is empty"
            }), 400
        
        # --- Extract material name if not provided ---
        if not material_name:
            material_name = extract_material_name_from_filename(file.filename)
        
        # --- Save temp PDF ---
        safe = secure_filename(file.filename) or "uploaded.pdf"
        temp_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{safe}"
        temp_path = TEMP_PDF_FOLDER / temp_filename
        
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)
        
        logging.info(f"Form upload: {file.filename} → {temp_filename}")
        
        # --- Run OCR ---
        logging.info(f"Running OCR on {file.filename}...")
        doc = fitz.open(temp_path)
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages)
        
        ocr_text_by_page = {}
        all_ocr_text = ""
        
        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                
                # Save to temp file for OCR
                timestamp_unique = int(time.time() * 1000) + page_num  # Unique per page
                temp_img_path = OUTPUT_FOLDER / f"temp_ocr_{page_num}_{timestamp_unique}.png"
                pix.save(str(temp_img_path))
                
                # Use existing OCR function that works
                ocr_text_list = run_paddle_ocr_on_file(temp_img_path)
                page_text = "\n".join(ocr_text_list)
                
                ocr_text_by_page[str(page_num + 1)] = page_text
                all_ocr_text += f"\n=== PAGE {page_num + 1} ===\n{page_text}"
                
                # Cleanup temp image
                temp_img_path.unlink(missing_ok=True)
                
            except Exception as e:
                logging.warning(f"OCR failed for page {page_num + 1}: {e}")
                ocr_text_by_page[str(page_num + 1)] = f"[OCR ERROR: {e}]"
        
        doc.close()
        logging.info(f"OCR completed for {pages_to_process} pages")
        
        # --- Use Groq to construct SpecificationsJson ---
        groq_result = process_ocr_with_groq(all_ocr_text, material_name)
        if not groq_result.get("success"):
            return jsonify({
                "success": False,
                "error": f"Groq processing failed: {groq_result.get('error', 'Unknown error')}"
            }), 500
        
        specs_json = groq_result.get("specifications", {})
        
        # Add raw excerpts
        if "raw_excerpts_by_page" not in specs_json:
            specs_json["raw_excerpts_by_page"] = {}
        specs_json["raw_excerpts_by_page"].update(ocr_text_by_page)
        
        # --- Validate SpecificationsJson ---
        if not isinstance(specs_json, dict):
            return jsonify({
                "success": False,
                "error": "Invalid specifications object from LLM"
            }), 500
        
        # --- Save to database ---
        matiere_id, fiche_id = save_material_and_fiche(
            material_name=material_name,
            type_matiere=type_matiere or material_name,
            specs_json=specs_json,
            source=source_type,
            reference=reference
        )
        
        logging.info(f"✅ Form submission processed: matiere_id={matiere_id}, fiche_id={fiche_id}")
        
        return jsonify({
            "success": True,
            "matiere_id": matiere_id,
            "fiche_id": fiche_id,
            "reference": reference,
            "source": source_type,
            "material_name": material_name,
            "ocr_pages": pages_to_process,
            "total_pages": total_pages,
            "specifications": specs_json
        }), 200
    
    except Exception as e:
        logging.error(f"Form processing error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
    finally:
        if 'temp_path' in locals() and temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    logging.info("Starting RFQ Processing API on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
