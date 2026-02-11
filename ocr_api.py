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

# --- CONFIGURATION ---
UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')

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

if __name__ == "__main__":
    logging.info("Starting RFQ Processing API on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
