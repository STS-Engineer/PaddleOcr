import logging
import time
from pathlib import Path
import contextlib
import os
import urllib.parse
from datetime import datetime, timedelta

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

import uuid
from threading import Thread
from flask import send_from_directory

# --- CONFIGURATION ---
UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"
client = OpenAI()

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

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "material_name": material_name,
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

    except Exception as e:
        logging.error(f"Save OCR Result Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def save_material_and_fiche(material_name: str, type_matiere: str, specs_json: dict, source: str, reference: str):
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT matiere_id FROM public.matieres WHERE nom_matiere = %s",
                    (material_name,)
                )
                row = cur.fetchone()

                if row:
                    matiere_id = row[0]
                    cur.execute(
                        """UPDATE public.matieres 
                           SET type_matiere = %s,
                               reference = %s,
                               date_mise_a_jour = NOW()
                           WHERE matiere_id = %s""",
                        (type_matiere, reference, matiere_id)
                    )
                else:
                    cur.execute(
                        """INSERT INTO public.matieres 
                           (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
                           VALUES (%s, %s, %s, NOW(), NOW())
                           RETURNING matiere_id""",
                        (material_name, type_matiere, reference)
                    )
                    matiere_id = cur.fetchone()[0]

                cur.execute(
                    """INSERT INTO public.fiches_matieres 
                       (matiere_id, date_creation_fiche, derniere_modification)
                       VALUES (%s, NOW(), NOW())
                       RETURNING fiche_id""",
                    (matiere_id,)
                )
                fiche_id = cur.fetchone()[0]

                cur.execute(
                    """INSERT INTO public.specifications 
                       (fiche_id, source_type, donnees, date_creation, derniere_modification)
                       VALUES (%s, %s, %s, NOW(), NOW())""",
                    (fiche_id, source, Json(specs_json))
                )

        return matiere_id, fiche_id
    finally:
        conn.close()

if __name__ == "__main__":
    logging.info("Starting RFQ Processing API on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
