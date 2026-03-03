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
import json

import uuid
from threading import Thread
from flask import send_from_directory

# --- CONFIGURATION ---

UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"
openai_api_key = (
    os.getenv("OPENAI_API_KEY")
)
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

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
        paddle_engine = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=True, 
            use_doc_unwarping=False,            
            use_textline_orientation=True     
        )
        
        doc_ori = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
        
        logging.info("Models initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL: Failed to load models: {e}")

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
        pass


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
    stem = Path(filename).stem
    parts = stem.split(' - ')
    if parts and parts[0].strip():
        ref = parts[0].strip()
        if ref.replace(' ', '').isdigit():
            return ref
    return None


def extract_reference_from_inspection_filename(filename: str) -> str:
    stem = Path(filename).stem
    ref = stem.strip()
    if ref.replace(' ', '').isdigit():
        return ref
    return None


@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_file(filename):
    try:
        return send_from_directory(str(TEMP_PDF_FOLDER), filename)
    except Exception:
        return jsonify({"success": False, "error": "temp_file_not_found"}), 404


@app.route("/upload-temp-pdf", methods=["POST"])
def upload_temp_pdf():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}

    refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
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
    try:
        return send_file(str(OUTPUT_FOLDER / secure_filename(filename)), mimetype='image/jpeg')
    except Exception:
        return jsonify({"success": False, "error": "Image not found"}), 404


@app.route('/download-image/<filename>', methods=['GET'])
def download_image(filename):
    try:
        return send_file(str(OUTPUT_FOLDER / secure_filename(filename)), as_attachment=True)
    except Exception:
        return jsonify({"success": False, "error": "Image not found"}), 404


@app.route('/process-rfq-id-to-images', methods=['POST'])
def process_rfq_id_to_images():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')
    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    mat = fitz.Matrix(2.0, 2.0)
    local_pdf_path = None
    conn = None
    download_url_page_1 = None

    try:
        logging.info(f"Fetching RFQ and searching for drawing: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()
        
        if not result or not result[0]:
            return jsonify({"success": False, "error": "RFQ ID or File Path not found"}), 404
        
        raw_paths = result[0].strip("{}").split(",")
        
        rfq_path_db = None
        for path in raw_paths:
            if 'drawing' in path.lower():
                rfq_path_db = path.strip()
                break
        
        if not rfq_path_db and raw_paths:
            rfq_path_db = raw_paths[0].strip()
        
        logging.info(f"Target file identified: {rfq_path_db}")
        
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

            raw_filename = f"{rfq_id}_page_{i+1}_{timestamp}.png"
            raw_path = OUTPUT_FOLDER / raw_filename
            pix.save(str(raw_path))

            upright_filename = f"{rfq_id}_page_{i+1}_{timestamp}_upright.png"
            upright_path = OUTPUT_FOLDER / upright_filename
            angle = save_upright_image(raw_path, upright_path)

            raw_path.unlink(missing_ok=True)

            logging.info(f"Running OCR on page {i+1} (angle={angle})...")
            ocr_text_list = run_paddle_ocr_on_file(upright_path)

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


@app.route("/process-openai-file-to-ocr", methods=["POST"])
def process_openai_file_to_ocr():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "JSON required. Provide 'openaiFileIdRefs', 'download_link' OR 'openai_file_id'"
        }), 400

    data = request.get_json()
    
    openai_file_id_refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    max_pages = int(data.get("max_pages", 20))
    original_filename = None

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
    download_url_page_1 = None

    try:
        pdf_bytes = None

        if download_link:
            logging.info(f"Downloading from link: {download_link[:100]}...")
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Download link failed (status={r.status_code}). Maybe expired?"
                }), 400
            pdf_bytes = r.content

            if not original_filename:
                try:
                    p = urllib.parse.urlparse(download_link).path
                    original_filename = Path(p).name or "uploaded.pdf"
                except Exception:
                    original_filename = "uploaded.pdf"

        if pdf_bytes is None and openai_file_id:
            logging.info(f"Using OpenAI file ID: {openai_file_id}")
            file_metadata = client.files.retrieve(openai_file_id)
            original_filename = getattr(file_metadata, "filename", None) or "uploaded.pdf"
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        safe_name = secure_filename(original_filename or "uploaded.pdf") or "uploaded.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        material_name = extract_material_name_from_filename(safe_name)

        unique_pdf_name = f"{uuid.uuid4().hex}_{timestamp}_{safe_name}"
        local_pdf_path = TEMP_PDF_FOLDER / unique_pdf_name

        with open(local_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        logging.info(f"PDF saved to: {local_pdf_path}")

        cleanup_old_files(OUTPUT_FOLDER)
        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        mat = fitz.Matrix(2.0, 2.0)
        
        base_url = request.host_url.rstrip('/')
        processed_pages = []

        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)

            filename = f"oa_page_{i+1}_{timestamp}.png"
            image_path = OUTPUT_FOLDER / filename
            pix.save(str(image_path))

            ocr_text_list = run_paddle_ocr_on_file(image_path)
            
            view_url = f"{base_url}/images/{filename}"
            dl_url   = f"{base_url}/download-image/{filename}"

            processed_pages.append({
                "page": i + 1,
                "url": view_url,
                "download_link_image": dl_url,
                "filename": filename,
                "ocr_text": ocr_text_list
            })
            
            if i == 0:
                download_url_page_1 = dl_url

        doc.close()

        reference = None
        if original_filename and ('SDS' in original_filename.upper() or 'MSDS' in original_filename.upper()):
            reference = extract_reference_from_msds_filename(original_filename)
            logging.info(f"Detected MSDS file. Extracted reference: {reference}")

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "material_name": material_name,
            "reference": reference,
            "total_pages": total_pages,
            "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "download_url_page_1_png": download_url_page_1,
            "pages": processed_pages
        }), 200

    except Exception as e:
        logging.error(f"PDF Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if local_pdf_path and Path(local_pdf_path).exists():
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
    
    openai_file_id_refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    max_pages = int(data.get("max_pages", 20))
    original_filename = None

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
        pdf_bytes = None

        if download_link:
            logging.info(f"Downloading from link: {download_link[:100]}...")
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": f"Download link failed (status={r.status_code}). Maybe expired?"
                }), 400
            pdf_bytes = r.content

            if not original_filename:
                try:
                    p = urllib.parse.urlparse(download_link).path
                    original_filename = Path(p).name or "uploaded.pdf"
                except Exception:
                    original_filename = "uploaded.pdf"

        if pdf_bytes is None and openai_file_id:
            logging.info(f"Using OpenAI file ID: {openai_file_id}")
            file_metadata = client.files.retrieve(openai_file_id)
            original_filename = getattr(file_metadata, "filename", None) or "uploaded.pdf"
            pdf_bytes = client.files.content(openai_file_id).read()

        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        safe_name = secure_filename(original_filename or "uploaded.pdf") or "uploaded.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        material_name = extract_material_name_from_filename(safe_name)

        unique_pdf_name = f"{uuid.uuid4().hex}_{timestamp}_{safe_name}"
        local_pdf_path = TEMP_PDF_FOLDER / unique_pdf_name

        with open(local_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        logging.info(f"PDF saved to: {local_pdf_path}")

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

        reference = None
        if original_filename and ('SDS' in original_filename.upper() or 'MSDS' in original_filename.upper()):
            reference = extract_reference_from_msds_filename(original_filename)
            logging.info(f"Detected MSDS file. Extracted reference: {reference}")

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "material_name": material_name,
            "reference": reference,
            "total_pages": total_pages,
            "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "pages": processed_pages
        }), 200

    except Exception as e:
        logging.error(f"PDF Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        pass


@app.route("/save-ocr-result", methods=["POST"])
def save_ocr_result():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}

    material_name = data.get("material_name")
    specifications = data.get("specifications")
    source = data.get("source_type", "datasheet")
    reference = data.get("reference")
    type_matiere = data.get("type_matiere") or material_name

    if isinstance(specifications, str):
        try:
            specifications = json.loads(specifications)
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "error": "Invalid JSON string in 'specifications'. Must be valid JSON."
            }), 400

    if not material_name:
        return jsonify({"success": False, "error": "Missing material_name"}), 400

    if not specifications:
        return jsonify({"success": False, "error": "Missing specifications JSON"}), 400

    try:
        matiere_id, fiche_id = save_material_and_fiche(
            material_name=material_name,
            type_matiere=type_matiere,
            specs_json=specifications,
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
        logging.error(f"Validation error: {ve}")
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Save OCR Result Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/specifications-by-reference", methods=["GET"])
def get_specifications_by_reference():
    reference = request.args.get("reference")
    if not reference:
        return jsonify({"success": False, "error": "Missing reference"}), 400

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
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}
    reference = data.get("reference")
    spec_id = data.get("spec_id")
    donnees = data.get("donnees")

    if isinstance(donnees, str):
        donnees = json.loads(donnees)

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
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                matiere_id = None

                if material_name:
                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE nom_matiere = %s",
                        (material_name,)
                    )
                    row = cur.fetchone()
                    if row:
                        matiere_id = row[0]
                        logging.info(f"Found by name (ID: {matiere_id}) → updating")
                        cur.execute(
                            """UPDATE public.matieres 
                               SET type_matiere = COALESCE(%s, type_matiere),
                                   reference = COALESCE(%s, reference),
                                   date_mise_a_jour = NOW()
                               WHERE matiere_id = %s""",
                            (type_matiere, reference, matiere_id)
                        )

                if not matiere_id and reference:
                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                        (reference,)
                    )
                    row = cur.fetchone()
                    if row:
                        matiere_id = row[0]
                        logging.info(f"Found by reference (ID: {matiere_id}) → updating")
                        cur.execute(
                            """UPDATE public.matieres 
                               SET nom_matiere = COALESCE(%s, nom_matiere),
                                   type_matiere = COALESCE(%s, type_matiere),
                                   date_mise_a_jour = NOW()
                               WHERE matiere_id = %s""",
                            (material_name, type_matiere, matiere_id)
                        )

                if not matiere_id:
                    logging.info(f"Not found → Creating new material: {material_name} ref={reference}")
                    cur.execute(
                        """INSERT INTO public.matieres 
                           (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
                           VALUES (%s, %s, %s, NOW(), NOW())
                           RETURNING matiere_id""",
                        (material_name, type_matiere, reference)
                    )
                    matiere_id = cur.fetchone()[0]
                    logging.info(f"Created new material with ID: {matiere_id}")

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
                    fiche_id, spec_id = existing
                    logging.info(f"Updating existing spec {spec_id}")
                    cur.execute(
                        """UPDATE public.specifications 
                           SET donnees = %s,
                               derniere_modification = NOW()
                           WHERE spec_id = %s""",
                        (Json(specs_json), spec_id)
                    )
                else:
                    cur.execute(
                        """INSERT INTO public.fiches_matieres 
                           (matiere_id, date_creation_fiche, derniere_modification)
                           VALUES (%s, NOW(), NOW())
                           RETURNING fiche_id""",
                        (matiere_id,)
                    )
                    fiche_id = cur.fetchone()[0]
                    logging.info(f"Created fiche with ID: {fiche_id}")

                    cur.execute(
                        """INSERT INTO public.specifications 
                           (fiche_id, source_type, donnees, date_creation, derniere_modification)
                           VALUES (%s, %s, %s, NOW(), NOW())""",
                        (fiche_id, source, Json(specs_json))
                    )
                    logging.info(f"Inserted specifications for fiche {fiche_id}")

        return matiere_id, fiche_id
    finally:
        conn.close()


def build_black_mix_adn_snapshot(cur, black_mix_id, product_reference, mix_name):
    """
    Build complete JSON snapshot (ADN) of Black Mix for archiving/export/PDF generation.
    Uses existing database cursor to gather all related data.
    """
    # Fetch all components with metadata
    cur.execute(
        """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                  m.reference, m.nom_matiere, c.metadata
           FROM public.black_mix_components c
           JOIN public.matieres m ON c.matiere_id = m.matiere_id
           WHERE c.black_mix_id = %s
           ORDER BY c.id""",
        (black_mix_id,)
    )
    components = [
        {
            "id": r[0],
            "component_name": r[1],
            "quantity": float(r[2]) if r[2] is not None else None,
            "unit": r[3],
            "reference": r[4],
            "material_name": r[5],
            "metadata": r[6]
        }
        for r in cur.fetchall()
    ]

    # Fetch all process steps
    cur.execute(
        """SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters,
                  ARRAY_AGG(m.reference ORDER BY m.reference) AS materials
           FROM public.black_mix_process_steps s
           LEFT JOIN public.black_mix_step_materials sm ON sm.process_step_id = s.id
           LEFT JOIN public.matieres m ON m.matiere_id = sm.matiere_id
           WHERE s.black_mix_id = %s
           GROUP BY s.id
           ORDER BY s.step_order""",
        (black_mix_id,)
    )
    process_steps = [
        {
            "step_order": r[1],
            "step_name": r[2],
            "machine": r[3],
            "parameters": r[4],
            "materials": list(r[5]) if r[5] and r[5][0] is not None else []
        }
        for r in cur.fetchall()
    ]

    # Build step_materials mapping
    step_materials = {}
    for step in process_steps:
        step_materials[str(step["step_order"])] = step["materials"]

    # Fetch control plan
    cur.execute(
        """SELECT parameter_name, target_value, min_value, max_value, unit
           FROM public.black_mix_control_plan
           WHERE black_mix_id = %s
           ORDER BY parameter_name""",
        (black_mix_id,)
    )
    control_plan = [
        {
            "parameter_name": r[0],
            "target_value": float(r[1]) if r[1] is not None else None,
            "min_value": float(r[2]) if r[2] is not None else None,
            "max_value": float(r[3]) if r[3] is not None else None,
            "unit": r[4]
        }
        for r in cur.fetchall()
    ]

    # Build complete ADN snapshot
    adn = {
        "black_mix_id": black_mix_id,
        "product_reference": product_reference,
        "mix_name": mix_name,
        "status": "draft",
        "created_at": datetime.now().isoformat(),
        "composition": components,
        "process_steps": process_steps,
        "step_materials": step_materials,
        "control_plan": control_plan,
        "snapshot_timestamp": datetime.now().isoformat()
    }
    
    return adn


# ==================== BLACK MIX ENDPOINTS ====================
# Structure réelle des tables :
#   black_mixes        : id, reference, name, status, created_at, updated_at
#   black_mix_components: id, black_mix_id, matiere_id, component_name, quantity_value, quantity_unit
#   black_mix_process_steps: id, black_mix_id, step_order, step_name, machine_name, parameters
#   black_mix_control_plan: id, black_mix_id, parameter_name, target_value, min_value, max_value, unit
# NOTE: black_mix_products n'existe pas — supprimé car redondant avec black_mixes.reference


@app.route("/black-mix/validate-material/<reference>", methods=["GET"])
def validate_black_mix_material(reference):
    """Validate if a material reference exists in the database."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT matiere_id, nom_matiere, reference FROM public.matieres WHERE reference = %s",
                (reference,)
            )
            row = cur.fetchone()
            
            if row:
                return jsonify({
                    "reference": reference,
                    "exists": True,
                    "material_name": row[1],
                    "matiere_id": row[0]
                }), 200
            else:
                return jsonify({
                    "reference": reference,
                    "exists": False,
                    "material_name": None,
                    "matiere_id": None
                }), 200
    except Exception as e:
        logging.error(f"Validate material error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/submit", methods=["POST"])
def submit_black_mix():
    """
    Submit a complete Black Mix with:
    - product_reference  → stored in black_mixes.reference
    - mix_name           → stored in black_mixes.name
    - components         → black_mix_components (component_name, quantity_value, quantity_unit, metadata)
    - process_steps      → black_mix_process_steps (step_order, step_name, machine_name, parameters)
    - control_plan       → black_mix_control_plan (parameter_name, target_value, min_value, max_value, unit)
    - document_revision_history → stored in black_mixes.document_revision_history (JSONB)
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}
    
    product_reference = data.get("product_reference")
    mix_name = data.get("mix_name")
    components = data.get("components", [])
    process_steps = data.get("process_steps", [])
    control_plan = data.get("control_plan", [])
    document_revision_history = data.get("document_revision_history")

    if not product_reference or not mix_name:
        return jsonify({
            "success": False,
            "error": "Missing required fields: product_reference, mix_name"
        }), 400

    conn = psycopg2.connect(DB_DSN)
    
    try:
        with conn:
            with conn.cursor() as cur:

                # --- Validate all material references first ---
                validation_errors = []
                for component in components:
                    ref = component.get("reference")
                    if not ref:
                        validation_errors.append("A component is missing its reference")
                        continue
                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                        (ref,)
                    )
                    if not cur.fetchone():
                        validation_errors.append(f"Invalid material reference: {ref}")
                
                if validation_errors:
                    return jsonify({
                        "success": False,
                        "message": "Validation errors found",
                        "validation_errors": validation_errors
                    }), 400

                # --- Create Black Mix with document revision history ---
                cur.execute(
                    """INSERT INTO public.black_mixes
                       (reference, name, status, created_at, document_revision_history)
                       VALUES (%s, %s, 'draft', NOW(), %s)
                       RETURNING id""",
                    (
                        product_reference, 
                        mix_name,
                        Json(document_revision_history) if document_revision_history else None
                    )
                )
                black_mix_id = cur.fetchone()[0]
                logging.info(f"Created Black Mix ID={black_mix_id} ref={product_reference}")
                if document_revision_history:
                    current_version = document_revision_history.get("current_version", "unknown")
                    logging.info(f"  └─ Document revision history: {current_version}")

                # --- Insert components ---
                for component in components:
                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                        (component["reference"],)
                    )
                    matiere_id = cur.fetchone()[0]

                    cur.execute(
                        """INSERT INTO public.black_mix_components
                           (black_mix_id, matiere_id, component_name, quantity_value, quantity_unit, metadata)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            black_mix_id,
                            matiere_id,
                            component.get("component_name") or component.get("reference"),
                            component.get("quantity"),
                            component.get("unit", "phr"),
                            Json(component.get("metadata", {}))
                        )
                    )

                # --- Insert process steps + step-materials ---
                for step in process_steps:

                    # Insert step and get its ID
                    cur.execute(
                        """INSERT INTO public.black_mix_process_steps
                           (black_mix_id, step_order, step_name, machine_name, parameters)
                           VALUES (%s, %s, %s, %s, %s)
                           RETURNING id""",
                        (
                            black_mix_id,
                            step.get("step_order"),
                            step.get("step_name"),
                            step.get("machine"),
                            Json(step.get("parameters", {}))
                        )
                    )

                    process_step_id = cur.fetchone()[0]

                    # Insert step-materials mapping
                    materials = step.get("materials", [])
                    for ref in materials:
                        cur.execute(
                            "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                            (ref,)
                        )
                        mat_row = cur.fetchone()
                        if not mat_row:
                            raise ValueError(f"Invalid material reference in step: {ref}")

                        matiere_id = mat_row[0]

                        cur.execute(
                            """INSERT INTO public.black_mix_step_materials
                               (process_step_id, matiere_id)
                               VALUES (%s, %s)
                               ON CONFLICT DO NOTHING""",
                            (process_step_id, matiere_id)
                        )

                # --- Insert control plan ---
                for param in control_plan:
                    cur.execute(
                        """INSERT INTO public.black_mix_control_plan
                           (black_mix_id, parameter_name, target_value, min_value, max_value, unit)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            black_mix_id,
                            param.get("parameter_name"),
                            param.get("target_value"),
                            param.get("min_value"),
                            param.get("max_value"),
                            param.get("unit")
                        )
                    )

                # --- Build and save Black Mix ADN (DNA/snapshot) ---
                adn_snapshot = build_black_mix_adn_snapshot(cur, black_mix_id, product_reference, mix_name)
                
                # Upsert into black_mix_adn table
                cur.execute(
                    """INSERT INTO public.black_mix_adn
                       (black_mix_id, adn_text, version, created_at)
                       VALUES (%s, %s, 1, NOW())
                       ON CONFLICT (black_mix_id) DO UPDATE
                       SET adn_text = EXCLUDED.adn_text,
                           version = black_mix_adn.version + 1
                       RETURNING id, version""",
                    (black_mix_id, Json(adn_snapshot))
                )
                adn_result = cur.fetchone()
                adn_id, adn_version = adn_result[0], adn_result[1] if adn_result else (None, 1)
                
                logging.info(f"✅ Black Mix ADN saved (ID={adn_id}, version={adn_version})")
                logging.info(f"✅ Black Mix '{mix_name}' saved successfully (ID={black_mix_id})")

                return jsonify({
                    "success": True,
                    "message": f"Black Mix '{mix_name}' created successfully",
                    "black_mix_id": black_mix_id,
                    "product_reference": product_reference,
                    "validation_errors": [],
                    "adn": {
                        "id": adn_id,
                        "version": adn_version,
                        "snapshot_created_at": datetime.now().isoformat()
                    }
                }), 200

    except Exception as e:
        logging.error(f"Submit Black Mix error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/list", methods=["GET"])
def list_black_mixes():
    """Get all Black Mixes."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, reference, name, status, created_at
                   FROM public.black_mixes
                   ORDER BY created_at DESC"""
            )
            rows = cur.fetchall()
            
            black_mixes = [
                {
                    "id": r[0],
                    "product_reference": r[1],
                    "mix_name": r[2],
                    "status": r[3],
                    "created_at": r[4].isoformat() if r[4] else None
                }
                for r in rows
            ]
            
            return jsonify({
                "success": True,
                "black_mixes": black_mixes
            }), 200

    except Exception as e:
        logging.error(f"List Black Mixes error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/<int:mix_id>", methods=["GET"])
def get_black_mix_details(mix_id):
    """Get complete details of a Black Mix."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:

            # --- Base info ---
            cur.execute(
                """SELECT id, reference, name, status, created_at, document_revision_history
                   FROM public.black_mixes
                   WHERE id = %s""",
                (mix_id,)
            )
            row = cur.fetchone()
            
            if not row:
                return jsonify({"success": False, "error": "Black Mix not found"}), 404
            
            result = {
                "id": row[0],
                "product_reference": row[1],
                "mix_name": row[2],
                "status": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "document_revision_history": row[5]
            }

            # --- Components ---
            cur.execute(
                """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                          m.reference, m.nom_matiere, c.metadata
                   FROM public.black_mix_components c
                   JOIN public.matieres m ON c.matiere_id = m.matiere_id
                   WHERE c.black_mix_id = %s""",
                (mix_id,)
            )
            result["components"] = [
                {
                    "id": r[0],
                    "component_name": r[1],
                    "quantity": float(r[2]) if r[2] is not None else None,
                    "unit": r[3],
                    "reference": r[4],
                    "material_name": r[5],
                    "metadata": r[6]
                }
                for r in cur.fetchall()
            ]

            # --- Process steps with materials ---
            cur.execute(
                """
                SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters,
                       ARRAY_AGG(m.reference ORDER BY m.reference) AS materials
                FROM public.black_mix_process_steps s
                LEFT JOIN public.black_mix_step_materials sm
                    ON sm.process_step_id = s.id
                LEFT JOIN public.matieres m
                    ON m.matiere_id = sm.matiere_id
                WHERE s.black_mix_id = %s
                GROUP BY s.id
                ORDER BY s.step_order
                """,
                (mix_id,)
            )
            result["process_steps"] = [
                {
                    "step_order": r[1],
                    "step_name": r[2],
                    "machine": r[3],
                    "parameters": r[4],
                    "materials": list(r[5]) if r[5] and r[5][0] is not None else []
                }
                for r in cur.fetchall()
            ]

            # --- Control plan ---
            cur.execute(
                """SELECT parameter_name, target_value, min_value, max_value, unit
                   FROM public.black_mix_control_plan
                   WHERE black_mix_id = %s""",
                (mix_id,)
            )
            result["control_plan"] = [
                {
                    "parameter_name": r[0],
                    "target_value": float(r[1]) if r[1] is not None else None,
                    "min_value": float(r[2]) if r[2] is not None else None,
                    "max_value": float(r[3]) if r[3] is not None else None,
                    "unit": r[4]
                }
                for r in cur.fetchall()
            ]

            return jsonify({
                "success": True,
                "black_mix": result
            }), 200

    except Exception as e:
        logging.error(f"Get Black Mix details error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/<int:mix_id>/adn", methods=["GET"])
def get_black_mix_adn(mix_id):
    """Retrieve the ADN (DNA/snapshot) of a Black Mix for export/PDF/archiving."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            # Get ADN from database
            cur.execute(
                """SELECT id, black_mix_id, adn_text, version, created_at
                   FROM public.black_mix_adn
                   WHERE black_mix_id = %s""",
                (mix_id,)
            )
            row = cur.fetchone()
            
            if not row:
                return jsonify({
                    "success": False,
                    "error": "ADN not found for this Black Mix"
                }), 404
            
            adn_id, black_mix_id, adn_text, version, created_at = row
            
            return jsonify({
                "success": True,
                "adn": {
                    "id": adn_id,
                    "black_mix_id": black_mix_id,
                    "version": version,
                    "created_at": created_at.isoformat() if created_at else None,
                    "snapshot": adn_text
                }
            }), 200

    except Exception as e:
        logging.error(f"Get Black Mix ADN error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "service": "ocr_api"}), 200


if __name__ == "__main__":
    logging.info("Starting RFQ Processing API on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
