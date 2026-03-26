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
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

TEMP_PDF_FOLDER = Path('/tmp/temp_pdfs')
TEMP_PDF_FOLDER.mkdir(exist_ok=True, parents=True)

DB_CONFIG = {
    "host": "avo-adb-002.postgres.database.azure.com",
    "database": "Costing_DB",
    "user": "administrationSTS",
    "password": "St$@0987"
}

GITHUB_OWNER = "STS-Engineer"
GITHUB_REPO = "RFQ-back"
GITHUB_BRANCH = "main"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)


# =====================================================================
# MIGRATION — run once at startup, idempotent
# =====================================================================

def ensure_sub_black_mix_column():
    """
    Adds sub_black_mix_id FK to black_mix_components and makes matiere_id
    nullable so a row can represent either a raw material OR a nested Black Mix.
    Also adds sub_black_mix_id to black_mix_step_materials so process steps
    can reference nested mixes too.
    Safe to call on every startup.
    """
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:

                # ── black_mix_components ──────────────────────────────────────

                # 1. Add sub_black_mix_id if missing
                cur.execute("""
                    ALTER TABLE public.black_mix_components
                    ADD COLUMN IF NOT EXISTS sub_black_mix_id INTEGER
                    REFERENCES public.black_mixes(id) ON DELETE SET NULL;
                """)

                # 2. Make matiere_id nullable
                cur.execute("""
                    SELECT is_nullable FROM information_schema.columns
                    WHERE table_schema='public'
                      AND table_name='black_mix_components'
                      AND column_name='matiere_id';
                """)
                row = cur.fetchone()
                if row and row[0] == 'NO':
                    cur.execute("""
                        ALTER TABLE public.black_mix_components
                        ALTER COLUMN matiere_id DROP NOT NULL;
                    """)
                    logging.info("Migration: black_mix_components.matiere_id made nullable")

                # 3. XOR constraint on black_mix_components
                cur.execute("""
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema='public'
                      AND table_name='black_mix_components'
                      AND constraint_name='chk_component_type';
                """)
                if not cur.fetchone():
                    cur.execute("""
                        ALTER TABLE public.black_mix_components
                        ADD CONSTRAINT chk_component_type CHECK (
                            (matiere_id IS NOT NULL AND sub_black_mix_id IS NULL)
                            OR
                            (matiere_id IS NULL AND sub_black_mix_id IS NOT NULL)
                        );
                    """)
                    logging.info("Migration: chk_component_type constraint added")

                # ── black_mix_step_materials ──────────────────────────────────

                # 4. Add sub_black_mix_id to step_materials if missing
                cur.execute("""
                    ALTER TABLE public.black_mix_step_materials
                    ADD COLUMN IF NOT EXISTS sub_black_mix_id INTEGER
                    REFERENCES public.black_mixes(id) ON DELETE SET NULL;
                """)

                # 5. Make matiere_id nullable in step_materials
                cur.execute("""
                    SELECT is_nullable FROM information_schema.columns
                    WHERE table_schema='public'
                      AND table_name='black_mix_step_materials'
                      AND column_name='matiere_id';
                """)
                row = cur.fetchone()
                if row and row[0] == 'NO':
                    cur.execute("""
                        ALTER TABLE public.black_mix_step_materials
                        ALTER COLUMN matiere_id DROP NOT NULL;
                    """)
                    logging.info("Migration: black_mix_step_materials.matiere_id made nullable")

                # 6. Update unique constraint to cover both columns
                cur.execute("""
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema='public'
                      AND table_name='black_mix_step_materials'
                      AND constraint_name='unique_step_matiere_v2';
                """)
                if not cur.fetchone():
                    # Drop old constraint if it exists
                    cur.execute("""
                        ALTER TABLE public.black_mix_step_materials
                        DROP CONSTRAINT IF EXISTS unique_step_matiere;
                    """)
                    cur.execute("""
                        ALTER TABLE public.black_mix_step_materials
                        ADD CONSTRAINT unique_step_matiere_v2
                        UNIQUE (process_step_id, matiere_id, sub_black_mix_id);
                    """)
                    logging.info("Migration: unique_step_matiere_v2 constraint added")

        logging.info("Migration ensure_sub_black_mix_column: OK")
    except Exception as e:
        logging.error(f"Migration failed: {e}", exc_info=True)
    finally:
        conn.close()

ensure_sub_black_mix_column()


# =====================================================================
# MODEL INITIALIZATION
# =====================================================================

paddle_engine = None
doc_ori = None

def init_models():
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


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def cleanup_old_files(folder, max_age_hours=24):
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
    while True:
        now = time.time()
        try:
            for f in TEMP_PDF_FOLDER.iterdir():
                if not f.is_file():
                    continue
                if now - f.stat().st_mtime > max_age_seconds:
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
            for t in (payload.get("rec_texts") or []):
                if t:
                    detected_texts.append(str(t))
        return detected_texts
    except Exception:
        logging.exception(f"PaddleOCR failed on {img_path}")
        return []


@app.route("/upload-temp-image", methods=["POST"])
def upload_temp_image():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}
    download_link, openai_file_id, original_name = extract_pdf_source(data)

    if not download_link and not openai_file_id:
        return jsonify({"success": False, "error": "Provide download_link or openai_file_id"}), 400

    try:
        image_bytes, original_name = fetch_pdf_bytes(
            download_link=download_link,
            openai_file_id=openai_file_id,
            original_name=original_name
        )
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded_image is None:
            return jsonify({"success": False, "error": "Downloaded file is not a valid image"}), 400

        safe_name = secure_filename(original_name) or "uploaded_image"
        image_stem = Path(safe_name).stem or "uploaded_image"
        timestamp = int(time.time())
        image_filename = f"{image_stem}_{timestamp}_{uuid.uuid4().hex}.jpg"
        image_path = OUTPUT_FOLDER / image_filename

        if not cv2.imwrite(str(image_path), decoded_image):
            return jsonify({"success": False, "error": "Failed to save image"}), 500

        Thread(target=delete_file_after_delay, args=(image_path,), daemon=True).start()

        return jsonify({
            "success": True,
            "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link),
            "original_filename": original_name,
            "image_filename": image_filename,
            "image_url": f"{request.host_url.rstrip('/')}/images/{image_filename}",
            "download_image_url": f"{request.host_url.rstrip('/')}/download-image/{image_filename}",
            "expires_in_seconds": TEMP_ENDPOINT_EXPIRY_SECONDS
        }), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logging.error(f"Temp image upload failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# =====================================================================
# NESTED BLACK MIX HELPERS
# =====================================================================

def resolve_component(cur, component: dict):
    """
    Determine whether a component is a raw material or a nested Black Mix.

    Resolution order:
      1. component_name matches black_mixes.name          → sub-mix
      2. reference      matches black_mixes.reference     → sub-mix
      3. reference      matches black_mixes.name (space-stripped) → sub-mix
      4. reference      matches matieres.reference        → raw material
      5. Otherwise                                        → unknown

    Returns:
      {
        "type":             "material" | "black_mix" | "unknown",
        "matiere_id":       int | None,
        "sub_black_mix_id": int | None,
        "resolved_name":    str,
      }
    """
    component_name = (component.get("component_name") or "").strip()
    reference = (component.get("reference") or "").strip()

    reference_no_spaces = reference.replace(" ", "").lower()
    component_name_no_spaces = component_name.replace(" ", "").lower()

    # 1. Match component_name against black_mixes.name
    if component_name:
        cur.execute(
            """SELECT id, name FROM public.black_mixes
               WHERE LOWER(TRIM(name)) = LOWER(%s)
                  OR LOWER(REPLACE(name, ' ', '')) = %s
               LIMIT 1""",
            (component_name, component_name_no_spaces)
        )
        row = cur.fetchone()
        if row:
            logging.info(f"  Component '{component_name}' → Black Mix id={row[0]} name='{row[1]}' (by name)")
            return {
                "type": "black_mix",
                "matiere_id": None,
                "sub_black_mix_id": row[0],
                "resolved_name": row[1]
            }

    # 2 & 3. Match reference against black_mixes.reference OR black_mixes.name (space-stripped)
    if reference:
        cur.execute(
            """SELECT id, name FROM public.black_mixes
               WHERE LOWER(TRIM(reference)) = LOWER(%s)
                  OR LOWER(REPLACE(name, ' ', '')) = %s
               LIMIT 1""",
            (reference, reference_no_spaces)
        )
        row = cur.fetchone()
        if row:
            logging.info(f"  Component ref='{reference}' → Black Mix id={row[0]} name='{row[1]}' (by ref/name)")
            return {
                "type": "black_mix",
                "matiere_id": None,
                "sub_black_mix_id": row[0],
                "resolved_name": row[1]
            }

    # 4. Match reference against matieres
    if reference:
        cur.execute(
            "SELECT matiere_id FROM public.matieres WHERE reference = %s LIMIT 1",
            (reference,)
        )
        row = cur.fetchone()
        if row:
            return {
                "type": "material",
                "matiere_id": row[0],
                "sub_black_mix_id": None,
                "resolved_name": component_name or reference
            }

    # 5. Not found
    return {
        "type": "unknown",
        "matiere_id": None,
        "sub_black_mix_id": None,
        "resolved_name": component_name or reference
    }


def resolve_step_material_ref(cur, ref: str):
    """
    Resolve a single reference from step_materials map.
    Returns the same structure as resolve_component.
    Checks black_mixes first, then matieres.
    """
    ref = (ref or "").strip()
    if not ref:
        return None

    ref_no_spaces = ref.replace(" ", "").lower()

    # Check black_mixes by reference or name
    cur.execute(
        """SELECT id, name FROM public.black_mixes
           WHERE LOWER(TRIM(reference)) = LOWER(%s)
              OR LOWER(REPLACE(name, ' ', '')) = %s
           LIMIT 1""",
        (ref, ref_no_spaces)
    )
    row = cur.fetchone()
    if row:
        return {"type": "black_mix", "matiere_id": None, "sub_black_mix_id": row[0]}

    # Check matieres
    cur.execute(
        "SELECT matiere_id FROM public.matieres WHERE reference = %s LIMIT 1",
        (ref,)
    )
    row = cur.fetchone()
    if row:
        return {"type": "material", "matiere_id": row[0], "sub_black_mix_id": None}

    return None


def expand_sub_black_mix(cur, sub_black_mix_id: int, visited: set = None) -> dict:
    """
    Recursively expand a nested Black Mix into a full snapshot dict.
    `visited` guards against circular references.
    """
    if visited is None:
        visited = set()

    if sub_black_mix_id in visited:
        logging.warning(f"Circular reference detected for black_mix_id={sub_black_mix_id}")
        return {"black_mix_id": sub_black_mix_id, "error": "circular_reference"}

    visited = visited | {sub_black_mix_id}

    cur.execute(
        "SELECT id, reference, name, status FROM public.black_mixes WHERE id = %s",
        (sub_black_mix_id,)
    )
    row = cur.fetchone()
    if not row:
        return {"black_mix_id": sub_black_mix_id, "error": "not_found"}

    snapshot = {
        "black_mix_id": row[0],
        "product_reference": row[1],
        "mix_name": row[2],
        "status": row[3],
        "components": []
    }

    cur.execute(
        """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                  c.matiere_id, c.sub_black_mix_id, c.metadata,
                  m.reference, m.nom_matiere
           FROM public.black_mix_components c
           LEFT JOIN public.matieres m ON m.matiere_id = c.matiere_id
           WHERE c.black_mix_id = %s
           ORDER BY c.id""",
        (sub_black_mix_id,)
    )
    for r in cur.fetchall():
        cid, cname, qty, unit, mat_id, sub_id, meta, mat_ref, mat_nom = r
        if sub_id is not None:
            snapshot["components"].append({
                "id": cid,
                "component_name": cname,
                "quantity": float(qty) if qty is not None else None,
                "unit": unit,
                "type": "black_mix",
                "sub_black_mix": expand_sub_black_mix(cur, sub_id, visited),
                "metadata": meta
            })
        else:
            snapshot["components"].append({
                "id": cid,
                "component_name": cname,
                "quantity": float(qty) if qty is not None else None,
                "unit": unit,
                "type": "material",
                "reference": mat_ref,
                "material_name": mat_nom,
                "metadata": meta
            })

    return snapshot


def build_black_mix_adn_snapshot(cur, black_mix_id, product_reference, mix_name):
    """Build complete JSON snapshot (ADN) — sub-mix components recursively expanded."""

    # ── Composition ──────────────────────────────────────────────────────────
    cur.execute(
        """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                  c.matiere_id, c.sub_black_mix_id, c.metadata,
                  m.reference, m.nom_matiere
           FROM public.black_mix_components c
           LEFT JOIN public.matieres m ON c.matiere_id = m.matiere_id
           WHERE c.black_mix_id = %s
           ORDER BY c.id""",
        (black_mix_id,)
    )
    components = []
    for r in cur.fetchall():
        cid, cname, qty, unit, mat_id, sub_id, meta, mat_ref, mat_nom = r
        if sub_id is not None:
            sub_snapshot = expand_sub_black_mix(cur, sub_id, visited={black_mix_id})
            components.append({
                "id": cid,
                "component_name": cname,
                "quantity": float(qty) if qty is not None else None,
                "unit": unit,
                "type": "black_mix",
                "sub_black_mix": sub_snapshot,
                "metadata": meta
            })
        else:
            components.append({
                "id": cid,
                "component_name": cname,
                "quantity": float(qty) if qty is not None else None,
                "unit": unit,
                "type": "material",
                "reference": mat_ref,
                "material_name": mat_nom,
                "metadata": meta
            })

    # ── Process steps — include both matiere refs and sub-mix refs ────────────
    cur.execute(
        """SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters
           FROM public.black_mix_process_steps s
           WHERE s.black_mix_id = %s
           ORDER BY s.step_order""",
        (black_mix_id,)
    )
    steps_raw = cur.fetchall()

    process_steps = []
    for s in steps_raw:
        step_id, step_order, step_name, machine, parameters = s

        # Fetch step materials: matieres refs + sub-mix refs
        cur.execute(
            """SELECT
                   sm.matiere_id,
                   sm.sub_black_mix_id,
                   m.reference         AS mat_ref,
                   bm.reference        AS sub_bm_ref
               FROM public.black_mix_step_materials sm
               LEFT JOIN public.matieres    m  ON m.matiere_id = sm.matiere_id
               LEFT JOIN public.black_mixes bm ON bm.id        = sm.sub_black_mix_id
               WHERE sm.process_step_id = %s""",
            (step_id,)
        )
        mat_refs = []
        for sm in cur.fetchall():
            ref = sm[3] if sm[1] is not None else sm[2]
            if ref:
                mat_refs.append(ref)

        process_steps.append({
            "step_order": step_order,
            "step_name": step_name,
            "machine": machine,
            "parameters": parameters,
            "materials": mat_refs,
        })

    # ── Control plan ─────────────────────────────────────────────────────────
    cur.execute(
        """SELECT parameter_name, target_value, min_value, max_value, unit
           FROM public.black_mix_control_plan
           WHERE black_mix_id = %s ORDER BY parameter_name""",
        (black_mix_id,)
    )
    control_plan = [
        {
            "parameter_name": r[0],
            "target_value": float(r[1]) if r[1] is not None else None,
            "min_value":    float(r[2]) if r[2] is not None else None,
            "max_value":    float(r[3]) if r[3] is not None else None,
            "unit": r[4]
        }
        for r in cur.fetchall()
    ]

    return {
        "black_mix_id": black_mix_id,
        "product_reference": product_reference,
        "mix_name": mix_name,
        "status": "draft",
        "created_at": datetime.now().isoformat(),
        "composition": components,
        "process_steps": process_steps,
        "step_materials": {str(s["step_order"]): s["materials"] for s in process_steps},
        "control_plan": control_plan,
        "snapshot_timestamp": datetime.now().isoformat()
    }


# =====================================================================
# MATERIAL DB HELPERS
# =====================================================================

def save_material_and_fiche(material_name, type_matiere, specs_json, source, reference=None):
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                matiere_id = None
                if material_name:
                    cur.execute("SELECT matiere_id FROM public.matieres WHERE nom_matiere = %s", (material_name,))
                    row = cur.fetchone()
                    if row:
                        matiere_id = row[0]
                        cur.execute(
                            """UPDATE public.matieres
                               SET type_matiere = COALESCE(%s, type_matiere),
                                   reference = COALESCE(%s, reference),
                                   date_mise_a_jour = NOW()
                               WHERE matiere_id = %s""",
                            (type_matiere, reference, matiere_id)
                        )

                if not matiere_id and reference:
                    cur.execute("SELECT matiere_id FROM public.matieres WHERE reference = %s", (reference,))
                    row = cur.fetchone()
                    if row:
                        matiere_id = row[0]
                        cur.execute(
                            """UPDATE public.matieres
                               SET nom_matiere = COALESCE(%s, nom_matiere),
                                   type_matiere = COALESCE(%s, type_matiere),
                                   date_mise_a_jour = NOW()
                               WHERE matiere_id = %s""",
                            (material_name, type_matiere, matiere_id)
                        )

                if not matiere_id:
                    cur.execute(
                        """INSERT INTO public.matieres
                           (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
                           VALUES (%s, %s, %s, NOW(), NOW()) RETURNING matiere_id""",
                        (material_name, type_matiere, reference)
                    )
                    matiere_id = cur.fetchone()[0]

                cur.execute(
                    """SELECT f.fiche_id, s.spec_id
                       FROM public.fiches_matieres f
                       JOIN public.specifications s ON s.fiche_id = f.fiche_id
                       WHERE f.matiere_id = %s AND s.source_type = %s
                       ORDER BY f.fiche_id DESC LIMIT 1""",
                    (matiere_id, source)
                )
                existing = cur.fetchone()
                if existing:
                    fiche_id, spec_id = existing
                    cur.execute(
                        """UPDATE public.specifications
                           SET donnees = %s, derniere_modification = NOW()
                           WHERE spec_id = %s""",
                        (Json(specs_json), spec_id)
                    )
                else:
                    cur.execute(
                        """INSERT INTO public.fiches_matieres
                           (matiere_id, date_creation_fiche, derniere_modification)
                           VALUES (%s, NOW(), NOW()) RETURNING fiche_id""",
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


# =====================================================================
# TEMP PDF / OCR ENDPOINTS
# =====================================================================

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
        if first_ref.get("download_link"):
            download_link = first_ref["download_link"]
        if first_ref.get("id") and not openai_file_id:
            openai_file_id = first_ref["id"]
        if first_ref.get("name"):
            original_name = first_ref["name"]

    if not download_link and not openai_file_id:
        return jsonify({"success": False, "error": "Provide download_link or openai_file_id"}), 400

    try:
        pdf_bytes = None
        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({"success": False, "error": f"Download failed (status={r.status_code})"}), 400
            pdf_bytes = r.content
        if pdf_bytes is None and openai_file_id:
            meta = client.files.retrieve(openai_file_id)
            if getattr(meta, "filename", None):
                original_name = meta.filename
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

        return jsonify({
            "success": True,
            "temp_filename": temp_filename,
            "temp_url": f"{request.host_url.rstrip('/')}/temp_files/{temp_filename}",
            "expires_in_seconds": 1800
        }), 200
    except Exception as e:
        logging.error(f"Temp upload failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload-temp-pdf-and-ocr", methods=["POST"])
def upload_temp_pdf_and_ocr():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400
    data = request.get_json(silent=True) or {}

    # ── openaiFileIdRefs (liste) ──────────────────────────────────────────────
    refs = data.get("openaiFileIdRefs")
    download_link = None
    openai_file_id = None
    original_name = "uploaded.pdf"

    if refs and isinstance(refs, list) and len(refs) > 0:
        first_ref = refs[0] if isinstance(refs[0], dict) else {"id": str(refs[0])}
        download_link  = first_ref.get("download_link")
        openai_file_id = first_ref.get("id")
        original_name  = first_ref.get("name") or "uploaded.pdf"
    # ─────────────────────────────────────────────────────────────────────────

    max_pages = int(data.get("max_pages", 20))

    if not download_link and not openai_file_id:
        return jsonify({"success": False, "error": "openaiFileIdRefs must contain 'id' or 'download_link'"}), 400

    timestamp = int(time.time())
    temp_path = None
    try:
        pdf_bytes = None
        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({"success": False, "error": f"Download failed (status={r.status_code})"}), 400
            pdf_bytes = r.content
        if pdf_bytes is None and openai_file_id:
            meta = client.files.retrieve(openai_file_id)
            if getattr(meta, "filename", None):
                original_name = meta.filename
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
            raw_path = OUTPUT_FOLDER / f"oa_raw_{i+1}_{timestamp}.png"
            pix.save(str(raw_path))
            upright_path = OUTPUT_FOLDER / f"oa_upright_{i+1}_{timestamp}.png"
            angle = save_upright_image(raw_path, upright_path)
            raw_path.unlink(missing_ok=True)
            processed_pages.append({
                "page": i+1,
                "rotation_angle": angle,
                "ocr_text": run_paddle_ocr_on_file(upright_path)
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
            "expires_in_seconds": 1800
        }), 200
    except Exception as e:
        logging.error(f"Upload+OCR failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/process-pdf-to-ocr", methods=["POST"])
def process_pdf_to_ocr():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400
    data = request.get_json()
    refs = data.get("openaiFileIdRefs")
    download_link = data.get("download_link")
    openai_file_id = data.get("openai_file_id")
    max_pages = int(data.get("max_pages", 20))
    original_filename = None

    if refs and isinstance(refs, list) and len(refs) > 0:
        download_link = refs[0].get("download_link")
        original_filename = refs[0].get("name", "uploaded.pdf")

    if not download_link and not openai_file_id:
        return jsonify({"success": False, "error": "Missing download_link or openai_file_id"}), 400

    timestamp = int(time.time())
    local_pdf_path = None
    try:
        pdf_bytes = None
        if download_link:
            r = requests.get(download_link, stream=True, timeout=60)
            if r.status_code != 200:
                return jsonify({"success": False, "error": f"Download failed (status={r.status_code})"}), 400
            pdf_bytes = r.content
            if not original_filename:
                try:
                    original_filename = Path(urllib.parse.urlparse(download_link).path).name or "uploaded.pdf"
                except Exception:
                    original_filename = "uploaded.pdf"
        if pdf_bytes is None and openai_file_id:
            meta = client.files.retrieve(openai_file_id)
            original_filename = getattr(meta, "filename", None) or "uploaded.pdf"
            pdf_bytes = client.files.content(openai_file_id).read()
        if pdf_bytes is None:
            return jsonify({"success": False, "error": "Unable to retrieve PDF bytes"}), 400

        safe_name = secure_filename(original_filename or "uploaded.pdf") or "uploaded.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        material_name = extract_material_name_from_filename(safe_name)
        local_pdf_path = TEMP_PDF_FOLDER / f"{uuid.uuid4().hex}_{timestamp}_{safe_name}"
        with open(local_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        cleanup_old_files(OUTPUT_FOLDER)
        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        mat = fitz.Matrix(2.0, 2.0)
        processed_pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=mat)
            raw_path = OUTPUT_FOLDER / f"oa_raw_{i+1}_{timestamp}.png"
            pix.save(str(raw_path))
            upright_path = OUTPUT_FOLDER / f"oa_upright_{i+1}_{timestamp}.png"
            angle = save_upright_image(raw_path, upright_path)
            raw_path.unlink(missing_ok=True)
            processed_pages.append({
                "page": i+1,
                "rotation_angle": angle,
                "ocr_text": run_paddle_ocr_on_file(upright_path)
            })
        doc.close()
        reference = None
        if original_filename and ('SDS' in original_filename.upper() or 'MSDS' in original_filename.upper()):
            reference = extract_reference_from_msds_filename(original_filename)
        return jsonify({
            "success": True, "openai_file_id": openai_file_id,
            "download_link_used": bool(download_link), "material_name": material_name,
            "reference": reference, "total_pages": total_pages,
            "converted_pages": len(processed_pages), "truncated": total_pages > max_pages,
            "pages": processed_pages
        }), 200
    except Exception as e:
        logging.error(f"PDF Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


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
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT rfq_file_path FROM public.main WHERE rfq_id = %s", (rfq_id,))
        result = cur.fetchone()
        if not result or not result[0]:
            return jsonify({"success": False, "error": "RFQ ID or File Path not found"}), 404

        raw_paths = result[0].strip("{}").split(",")
        rfq_path_db = next((p.strip() for p in raw_paths if 'drawing' in p.lower()), raw_paths[0].strip())
        clean_path = rfq_path_db.strip("/")
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{urllib.parse.quote(clean_path)}"
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub error: {resp.status_code}"}), 400

        local_pdf_path = UPLOAD_FOLDER / f"{rfq_id}_{int(time.time())}.pdf"
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
            raw_path = OUTPUT_FOLDER / f"{rfq_id}_page_{i+1}_{timestamp}.png"
            pix.save(str(raw_path))
            upright_path = OUTPUT_FOLDER / f"{rfq_id}_page_{i+1}_{timestamp}_upright.png"
            angle = save_upright_image(raw_path, upright_path)
            raw_path.unlink(missing_ok=True)
            upright_filename = upright_path.name
            view_url = f"{base_url}/images/{upright_filename}"
            dl_url = f"{base_url}/download-image/{upright_filename}"
            processed_pages.append({
                "page": i+1, "url": view_url,
                "lien_pour_telecharger_l_image": dl_url,
                "filename": upright_filename, "rotation_angle": angle,
                "ocr_text": run_paddle_ocr_on_file(upright_path)
            })
            if i == 0:
                download_url_page_1 = dl_url
        doc.close()
        if not download_url_page_1 and processed_pages:
            download_url_page_1 = processed_pages[0]["lien_pour_telecharger_l_image"]
        return jsonify({
            "success": True, "rfq_id": rfq_id, "source_path": rfq_path_db,
            "total_pages": total_pages, "converted_pages": len(processed_pages),
            "truncated": total_pages > max_pages,
            "download_url_page_1_png": download_url_page_1, "images": processed_pages
        }), 200
    except Exception as e:
        logging.error(f"RFQ Process Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if conn:
            with contextlib.suppress(Exception):
                cur.close(); conn.close()
        if local_pdf_path and local_pdf_path.exists():
            with contextlib.suppress(Exception):
                os.remove(local_pdf_path)


# =====================================================================
# SPECIFICATIONS ENDPOINTS
# =====================================================================

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
            return jsonify({"success": False, "error": "Invalid JSON in 'specifications'"}), 400
    if not material_name:
        return jsonify({"success": False, "error": "Missing material_name"}), 400
    if not specifications:
        return jsonify({"success": False, "error": "Missing specifications JSON"}), 400
    try:
        matiere_id, fiche_id = save_material_and_fiche(
            material_name, type_matiere, specifications, source, reference
        )
        return jsonify({
            "success": True, "matiere_id": matiere_id, "fiche_id": fiche_id,
            "source": source, "reference": reference
        }), 200
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
                    """SELECT s.spec_id, s.fiche_id, s.source_type, s.donnees
                       FROM public.matieres m
                       JOIN public.fiches_matieres f ON f.matiere_id = m.matiere_id
                       JOIN public.specifications s ON s.fiche_id = f.fiche_id
                       WHERE m.reference = %s ORDER BY s.spec_id ASC""",
                    (reference,)
                )
                specs = [
                    {"spec_id": r[0], "fiche_id": r[1], "source_type": r[2], "donnees": r[3]}
                    for r in cur.fetchall()
                ]
        return jsonify({"success": True, "reference": reference, "specifications": specs}), 200
    except Exception as e:
        logging.error(f"Get specifications error: {e}", exc_info=True)
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
                        """SELECT m.matiere_id, f.fiche_id, s.spec_id
                           FROM public.matieres m
                           JOIN public.fiches_matieres f ON f.matiere_id = m.matiere_id
                           JOIN public.specifications s ON s.fiche_id = f.fiche_id
                           WHERE m.reference = %s AND s.spec_id = %s LIMIT 1""",
                        (reference, spec_id)
                    )
                else:
                    cur.execute(
                        """SELECT f.matiere_id, s.fiche_id, s.spec_id
                           FROM public.specifications s
                           JOIN public.fiches_matieres f ON f.fiche_id = s.fiche_id
                           WHERE s.spec_id = %s LIMIT 1""",
                        (spec_id,)
                    )
                row = cur.fetchone()
                if not row:
                    return jsonify({"success": False, "error": "Specification not found"}), 404
                matiere_id, fiche_id, spec_id = row
                cur.execute(
                    "UPDATE public.specifications SET donnees = %s, derniere_modification = NOW() WHERE spec_id = %s",
                    (Json(donnees), spec_id)
                )
        return jsonify({
            "success": True, "matiere_id": matiere_id,
            "fiche_id": fiche_id, "spec_id": spec_id, "reference": reference
        }), 200
    except Exception as e:
        logging.error(f"Update specification error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


# =====================================================================
# BLACK MIX ENDPOINTS
# =====================================================================

@app.route("/black-mix/resolve-component", methods=["GET"])
def resolve_component_endpoint():
    """
    Utility: given component_name and/or reference, returns whether it
    resolves to a raw material or a nested Black Mix.
    """
    component_name = (request.args.get("component_name") or "").strip()
    reference = (request.args.get("reference") or "").strip()
    if not component_name and not reference:
        return jsonify({"success": False, "error": "Provide component_name and/or reference"}), 400

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            result = resolve_component(cur, {"component_name": component_name, "reference": reference})
        return jsonify({"success": True, "resolution": result}), 200
    except Exception as e:
        logging.error(f"Resolve component error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/validate-material/<reference>", methods=["GET"])
def validate_black_mix_material(reference):
    """
    Validate if a reference exists in matieres OR as a Black Mix.
    Returns type: 'material' | 'black_mix' | null
    """
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            # Check matieres first
            cur.execute(
                "SELECT matiere_id, nom_matiere FROM public.matieres WHERE reference = %s",
                (reference,)
            )
            mat_row = cur.fetchone()
            if mat_row:
                return jsonify({
                    "reference": reference, "exists": True, "type": "material",
                    "material_name": mat_row[1], "matiere_id": mat_row[0]
                }), 200

            # Check black_mixes
            ref_no_spaces = reference.replace(" ", "").lower()
            cur.execute(
                """SELECT id, name, reference FROM public.black_mixes
                   WHERE LOWER(TRIM(reference)) = LOWER(%s)
                      OR LOWER(REPLACE(name, ' ', '')) = %s
                   LIMIT 1""",
                (reference, ref_no_spaces)
            )
            mix_row = cur.fetchone()
            if mix_row:
                return jsonify({
                    "reference": reference, "exists": True, "type": "black_mix",
                    "material_name": mix_row[1], "black_mix_id": mix_row[0],
                    "product_reference": mix_row[2]
                }), 200

            return jsonify({
                "reference": reference, "exists": False,
                "type": None, "material_name": None
            }), 200
    except Exception as e:
        logging.error(f"Validate material error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/submit", methods=["POST"])
def submit_black_mix():
    """
    Submit a complete Black Mix.

    Components are resolved automatically:
      - component_name matches black_mixes.name → stored as sub-mix
      - reference matches black_mixes.reference → stored as sub-mix
      - reference matches matieres.reference    → stored as raw material
      - Otherwise                               → validation error

    step_materials references are also resolved against BOTH tables
    (matieres and black_mixes), and deduplicated per step to avoid
    unique constraint violations on black_mix_step_materials.
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON required"}), 400

    data = request.get_json(silent=True) or {}
    product_reference         = data.get("product_reference")
    mix_name                  = data.get("mix_name")
    components                = data.get("components", [])
    process_steps             = data.get("process_steps", [])
    step_materials_map        = data.get("step_materials", {})
    control_plan              = data.get("control_plan", [])
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

                # ── 1. Resolve & validate all components ──────────────────────
                validation_errors = []
                resolved_components = []

                for component in components:
                    resolution = resolve_component(cur, component)
                    if resolution["type"] == "unknown":
                        label = component.get("component_name") or component.get("reference") or "?"
                        validation_errors.append(f"'{label}' not found in materials or Black Mixes")
                    else:
                        resolved_components.append({**component, **resolution})

                if validation_errors:
                    return jsonify({
                        "success": False,
                        "message": "Validation errors found",
                        "validation_errors": validation_errors
                    }), 400

                # ── 2. Create Black Mix record ─────────────────────────────────
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

                # ── 3. Insert components (material or nested Black Mix) ────────
                for comp in resolved_components:
                    if comp["type"] == "black_mix":
                        cur.execute(
                            """INSERT INTO public.black_mix_components
                               (black_mix_id, sub_black_mix_id, component_name,
                                quantity_value, quantity_unit, metadata)
                               VALUES (%s, %s, %s, %s, %s, %s)""",
                            (
                                black_mix_id,
                                comp["sub_black_mix_id"],
                                comp.get("component_name") or comp["resolved_name"],
                                comp.get("quantity"),
                                comp.get("unit", "phr"),
                                Json(comp.get("metadata", {}))
                            )
                        )
                        logging.info(
                            f"  └─ Nested Black Mix '{comp['resolved_name']}' "
                            f"(id={comp['sub_black_mix_id']}) inserted as component"
                        )
                    else:
                        cur.execute(
                            """INSERT INTO public.black_mix_components
                               (black_mix_id, matiere_id, component_name,
                                quantity_value, quantity_unit, metadata)
                               VALUES (%s, %s, %s, %s, %s, %s)""",
                            (
                                black_mix_id,
                                comp["matiere_id"],
                                comp.get("component_name") or comp.get("reference"),
                                comp.get("quantity"),
                                comp.get("unit", "phr"),
                                Json(comp.get("metadata", {}))
                            )
                        )

                logging.info(f"  └─ {len(resolved_components)} components inserted")

                # ── 4. Insert process steps + step materials ───────────────────
                #
                # KEY FIX: step_materials refs are resolved against BOTH
                # black_mixes AND matieres, and deduplicated per step
                # (same matiere_id or sub_black_mix_id only once per step_id)
                # to avoid violating unique_step_matiere constraint.
                #
                total_step_materials = 0
                for step in process_steps:
                    cur.execute(
                        """INSERT INTO public.black_mix_process_steps
                           (black_mix_id, step_order, step_name, machine_name, parameters)
                           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                        (
                            black_mix_id,
                            step.get("step_order"),
                            step.get("step_name"),
                            step.get("machine"),
                            Json(step.get("parameters", {}))
                        )
                    )
                    process_step_id = cur.fetchone()[0]

                    step_order_key = str(step.get("step_order"))
                    refs_for_step = step_materials_map.get(step_order_key, [])
                    logging.info(
                        f"  └─ Step {step_order_key} '{step.get('step_name')}' "
                        f"→ {len(refs_for_step)} ref(s)"
                    )

                    # Deduplicate by (type, id) to prevent unique constraint violation.
                    # This handles cases like 6600071 appearing twice in the same step
                    # (e.g. once as "Lösung" and once as "Spülen") — they share the
                    # same matiere_id so only the first occurrence is inserted.
                    seen_step_ids: set = set()

                    for ref in refs_for_step:
                        if not ref:
                            continue

                        resolved = resolve_step_material_ref(cur, ref)

                        if resolved is None:
                            logging.warning(
                                f"    ⚠️  step_materials ref '{ref}' not found "
                                f"in matieres or black_mixes, skipping"
                            )
                            continue

                        # Build dedup key: ("material", matiere_id) or ("black_mix", bm_id)
                        if resolved["type"] == "material":
                            dedup_key = ("material", resolved["matiere_id"])
                        else:
                            dedup_key = ("black_mix", resolved["sub_black_mix_id"])

                        if dedup_key in seen_step_ids:
                            logging.debug(
                                f"    ↩  Skipping duplicate step-material: "
                                f"ref='{ref}' key={dedup_key}"
                            )
                            continue

                        seen_step_ids.add(dedup_key)

                        if resolved["type"] == "material":
                            cur.execute(
                                """INSERT INTO public.black_mix_step_materials
                                   (process_step_id, matiere_id)
                                   VALUES (%s, %s)""",
                                (process_step_id, resolved["matiere_id"])
                            )
                        else:
                            # Sub-black-mix in a step
                            cur.execute(
                                """INSERT INTO public.black_mix_step_materials
                                   (process_step_id, sub_black_mix_id)
                                   VALUES (%s, %s)""",
                                (process_step_id, resolved["sub_black_mix_id"])
                            )

                        total_step_materials += 1

                logging.info(f"  └─ {total_step_materials} step-material links inserted")

                # ── 5. Insert control plan ─────────────────────────────────────
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
                logging.info(f"  └─ {len(control_plan)} control plan parameters inserted")

                # ── 6. Build and save ADN snapshot ────────────────────────────
                adn_snapshot = build_black_mix_adn_snapshot(
                    cur, black_mix_id, product_reference, mix_name
                )

                cur.execute(
                    """SELECT id, version FROM public.black_mix_adn
                       WHERE black_mix_id = %s ORDER BY id DESC LIMIT 1""",
                    (black_mix_id,)
                )
                existing_adn = cur.fetchone()

                if existing_adn:
                    existing_adn_id, existing_version = existing_adn
                    next_version = (existing_version or 0) + 1
                    cur.execute(
                        """UPDATE public.black_mix_adn
                           SET adn_text = %s, version = %s, created_at = NOW()
                           WHERE id = %s RETURNING id, version""",
                        (Json(adn_snapshot), next_version, existing_adn_id)
                    )
                else:
                    cur.execute(
                        """INSERT INTO public.black_mix_adn
                           (black_mix_id, adn_text, version, created_at)
                           VALUES (%s, %s, 1, NOW()) RETURNING id, version""",
                        (black_mix_id, Json(adn_snapshot))
                    )

                adn_result = cur.fetchone()
                adn_id, adn_version = (adn_result[0], adn_result[1]) if adn_result else (None, None)
                logging.info(f"✅ Black Mix '{mix_name}' saved (ID={black_mix_id}, ADN v{adn_version})")

                return jsonify({
                    "success": True,
                    "message": f"Black Mix '{mix_name}' created successfully",
                    "black_mix_id": black_mix_id,
                    "product_reference": product_reference,
                    "components_summary": [
                        {
                            "component_name": c.get("component_name") or c.get("reference"),
                            "type": c["type"],
                            "resolved_as": c["resolved_name"],
                            "id": c.get("sub_black_mix_id") or c.get("matiere_id")
                        }
                        for c in resolved_components
                    ],
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
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, reference, name, status, created_at FROM public.black_mixes ORDER BY created_at DESC"
            )
            black_mixes = [
                {
                    "id": r[0], "product_reference": r[1], "mix_name": r[2],
                    "status": r[3], "created_at": r[4].isoformat() if r[4] else None
                }
                for r in cur.fetchall()
            ]
        return jsonify({"success": True, "black_mixes": black_mixes}), 200
    except Exception as e:
        logging.error(f"List Black Mixes error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/<int:mix_id>", methods=["GET"])
def get_black_mix_details(mix_id):
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, reference, name, status, created_at, document_revision_history
                   FROM public.black_mixes WHERE id = %s""",
                (mix_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"success": False, "error": "Black Mix not found"}), 404

            result = {
                "id": row[0], "product_reference": row[1], "mix_name": row[2],
                "status": row[3], "created_at": row[4].isoformat() if row[4] else None,
                "document_revision_history": row[5]
            }

            # Components — with sub-mix awareness
            cur.execute(
                """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                          c.matiere_id, c.sub_black_mix_id, c.metadata,
                          m.reference, m.nom_matiere,
                          bm.reference AS sub_mix_ref, bm.name AS sub_mix_name
                   FROM public.black_mix_components c
                   LEFT JOIN public.matieres m ON c.matiere_id = m.matiere_id
                   LEFT JOIN public.black_mixes bm ON c.sub_black_mix_id = bm.id
                   WHERE c.black_mix_id = %s
                   ORDER BY c.id""",
                (mix_id,)
            )
            components = []
            for r in cur.fetchall():
                cid, cname, qty, unit, mat_id, sub_id, meta, mat_ref, mat_nom, sub_ref, sub_name = r
                if sub_id is not None:
                    components.append({
                        "id": cid, "component_name": cname,
                        "quantity": float(qty) if qty is not None else None,
                        "unit": unit, "type": "black_mix",
                        "sub_black_mix_id": sub_id,
                        "sub_black_mix_reference": sub_ref,
                        "sub_black_mix_name": sub_name,
                        "metadata": meta
                    })
                else:
                    components.append({
                        "id": cid, "component_name": cname,
                        "quantity": float(qty) if qty is not None else None,
                        "unit": unit, "type": "material",
                        "reference": mat_ref, "material_name": mat_nom,
                        "metadata": meta
                    })
            result["components"] = components

            # Process steps — with sub-mix refs in materials list
            cur.execute(
                """SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters
                   FROM public.black_mix_process_steps s
                   WHERE s.black_mix_id = %s
                   ORDER BY s.step_order""",
                (mix_id,)
            )
            steps_raw = cur.fetchall()
            process_steps = []
            for s in steps_raw:
                step_id, step_order, step_name, machine, parameters = s
                cur.execute(
                    """SELECT sm.matiere_id, sm.sub_black_mix_id,
                              m.reference AS mat_ref, bm.reference AS sub_bm_ref
                       FROM public.black_mix_step_materials sm
                       LEFT JOIN public.matieres m ON m.matiere_id = sm.matiere_id
                       LEFT JOIN public.black_mixes bm ON bm.id = sm.sub_black_mix_id
                       WHERE sm.process_step_id = %s""",
                    (step_id,)
                )
                mat_refs = []
                for sm in cur.fetchall():
                    ref = sm[3] if sm[1] is not None else sm[2]
                    if ref:
                        mat_refs.append(ref)
                process_steps.append({
                    "step_order": step_order, "step_name": step_name,
                    "machine": machine, "parameters": parameters,
                    "materials": mat_refs
                })
            result["process_steps"] = process_steps

            # Control plan
            cur.execute(
                """SELECT parameter_name, target_value, min_value, max_value, unit
                   FROM public.black_mix_control_plan WHERE black_mix_id = %s""",
                (mix_id,)
            )
            result["control_plan"] = [
                {
                    "parameter_name": r[0],
                    "target_value": float(r[1]) if r[1] is not None else None,
                    "min_value":    float(r[2]) if r[2] is not None else None,
                    "max_value":    float(r[3]) if r[3] is not None else None,
                    "unit": r[4]
                }
                for r in cur.fetchall()
            ]

        return jsonify({"success": True, "black_mix": result}), 200
    except Exception as e:
        logging.error(f"Get Black Mix details error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/<int:mix_id>/adn", methods=["GET"])
def get_black_mix_adn(mix_id):
    """Retrieve the ADN snapshot — sub-mixes are fully expanded."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, black_mix_id, adn_text, version, created_at
                   FROM public.black_mix_adn WHERE black_mix_id = %s
                   ORDER BY version DESC LIMIT 1""",
                (mix_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"success": False, "error": "ADN not found for this Black Mix"}), 404
            adn_id, bm_id, adn_text, version, created_at = row
        return jsonify({
            "success": True,
            "adn": {
                "id": adn_id, "black_mix_id": bm_id, "version": version,
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
    return jsonify({"status": "ok", "service": "ocr_api"}), 200


if __name__ == "__main__":
    logging.info("Starting RFQ Processing API on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
