"""
geoRemove Backend Server
========================
Pakai: pip install flask flask-cors rembg[gpu] Pillow onnxruntime
Untuk CPU only: pip install flask flask-cors rembg Pillow onnxruntime

Jalankan: python georemove_server.py
Port: 5050
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io
import base64
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("georemove")

app = Flask(__name__)
CORS(app, origins="*")  # Ganti dengan domain kamu di produksi

# ── load model sekali saat startup ────────────────────────────────────────────
# Model options (dari yang paling akurat):
#   "isnet-general-use"  → paling akurat, terbaik untuk foto umum
#   "u2net"             → cepat, akurat untuk orang/produk  
#   "u2net_human_seg"   → khusus manusia/portrait (bagus untuk foto KTP/pas foto)
#   "silueta"           → ringan, cepat
#   "birefnet-general"  → state of the art terbaru
MODEL_NAME = os.getenv("REMBG_MODEL", "isnet-general-use")

log.info(f"Memuat model: {MODEL_NAME} ...")
session = new_session(MODEL_NAME)
log.info(f"Model {MODEL_NAME} siap.")

MAX_SIZE_MB = 20
MAX_DIM = 4096  # max dimension sebelum resize untuk hemat memori


def limit_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > MAX_DIM or h > MAX_DIM:
        ratio = min(MAX_DIM / w, MAX_DIM / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    try:
        # ── terima file ───────────────────────────────────────────────────────
        if "file" in request.files:
            file = request.files["file"]
            if file.content_length and file.content_length > MAX_SIZE_MB * 1024 * 1024:
                return jsonify({"error": f"File terlalu besar (max {MAX_SIZE_MB}MB)"}), 413
            img_bytes = file.read()

        elif request.is_json and "image" in request.json:
            # base64 fallback
            b64 = request.json["image"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)

        else:
            return jsonify({"error": "Kirim file via multipart/form-data (field: file) atau JSON {image: base64}"}), 400

        # ── buka & proses ─────────────────────────────────────────────────────
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        orig_size = img.size
        img = limit_size(img)

        log.info(f"Proses gambar {orig_size} → {img.size}")

        # alpha_matting=True → tepi lebih halus (rambut, bulu)
        result: Image.Image = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )

        # kembalikan ke ukuran asli jika di-resize
        if result.size != orig_size:
            # rebuild dari original bytes dengan mask baru
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            mask = result.getchannel("A").resize(orig_size, Image.LANCZOS)
            orig_img.putalpha(mask)
            result = orig_img

        # ── encode output ─────────────────────────────────────────────────────
        out_format = request.args.get("format", "png").lower()
        buf = io.BytesIO()

        if out_format == "webp":
            result.save(buf, format="WEBP", lossless=True, quality=100)
            mime = "image/webp"
        elif out_format == "jpg":
            # JPG tidak support alpha → composite ke putih
            bg = request.args.get("bg", "white")
            if bg.startswith("#"):
                r = int(bg[1:3], 16); g = int(bg[3:5], 16); b = int(bg[5:7], 16)
                bg_img = Image.new("RGB", result.size, (r, g, b))
            else:
                bg_img = Image.new("RGB", result.size, (255, 255, 255))
            bg_img.paste(result, mask=result.getchannel("A"))
            bg_img.save(buf, format="JPEG", quality=97, subsampling=0)
            mime = "image/jpeg"
        else:  # default PNG
            result.save(buf, format="PNG", optimize=False)
            mime = "image/png"

        buf.seek(0)
        size_kb = buf.getbuffer().nbytes // 1024
        log.info(f"Output: {out_format.upper()} {result.size[0]}x{result.size[1]} ({size_kb}KB)")

        return send_file(buf, mimetype=mime, as_attachment=False)

    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/remove-bg-b64", methods=["POST"])
def remove_bg_b64():
    """Endpoint alternatif: terima & kembalikan base64 JSON (untuk CORS sederhana)"""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Field 'image' (base64) wajib diisi"}), 400

        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        orig_size = img.size
        img_proc = limit_size(img)

        result = remove(
            img_proc,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )

        if result.size != orig_size:
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            mask = result.getchannel("A").resize(orig_size, Image.LANCZOS)
            orig_img.putalpha(mask)
            result = orig_img

        buf = io.BytesIO()
        result.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{encoded}",
            "width": result.size[0],
            "height": result.size[1],
        })

    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    log.info(f"geoRemove server berjalan di port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
