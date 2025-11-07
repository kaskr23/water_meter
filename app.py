import os
import csv
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    Response,
)

import numpy as np
import cv2
import pytesseract

app = Flask(__name__)

# Paths inside the container (ephemeral, reset on redeploy/restart)
LAST_IMG_PATH = "/tmp/last_upload.jpg"
READINGS_CSV = "/tmp/readings.csv"
IMAGE_LOG_CSV = "/tmp/image_log.csv"

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
print("✅ Using tesseract at:", pytesseract.pytesseract.tesseract_cmd)


@app.route("/")
def index():
    """Main page: show readings table & link to last image."""
    rows = []
    if os.path.exists(READINGS_CSV):
        with open(READINGS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Water Meter Readings</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; padding: 20px; }",
        "table { border-collapse: collapse; width: 100%; max-width: 700px; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
        "th { background: #f0f0f0; }",
        ".no-data { color: #888; }",
        "a { color: #007bff; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "</style></head><body>",
        "<h1>💧 Water Meter Readings</h1>",
        "<p><a href='/last' target='_blank'>View last image</a></p>",
    ]

    if not rows:
        html_parts.append("<p class='no-data'>No readings yet.</p>")
    else:
        html_parts.append("<table>")
        html_parts.append("<tr><th>#</th><th>Timestamp (UTC)</th><th>Reading</th><th>Source IP</th></tr>")
        for i, row in enumerate(reversed(rows), start=1):  # newest first
            html_parts.append(
                f"<tr><td>{i}</td><td>{row['timestamp']}</td>"
                f"<td>{row['reading']}</td><td>{row['ip']}</td></tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")
    return Response("\n".join(html_parts), mimetype="text/html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """ESP32-CAM sends image here via POST (Content-Type: image/jpeg)."""
    if request.method == "GET":
        return "✅ /upload is ready (send POST with image/jpeg)", 200

    try:
        img_bytes = request.data
        if not img_bytes:
            return jsonify({"error": "empty body"}), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "invalid image"}), 400

        # Save last image
        cv2.imwrite(LAST_IMG_PATH, img)

        h, w, _ = img.shape
        size = len(img_bytes)
        ip = request.remote_addr or "unknown"
        ts = datetime.utcnow().isoformat()

        print(f"📸 Received image {w}x{h}, {size} bytes from {ip} at {ts}")

        append_image_log(ts, w, h, size, ip)

        # ---- OCR PART ----
        reading = run_ocr_on_image(img)
        if reading:
            print(f"✅ OCR reading: {reading}")
            append_reading(ts, reading, ip)
        else:
            print("⚠️ OCR: no digits detected")

        return jsonify({
            "status": "ok",
            "timestamp": ts,
            "width": w,
            "height": h,
            "size": size,
            "ip": ip,
            "reading": reading,
        }), 200

    except Exception as e:
        print("❌ Error in /upload:", e)
        return jsonify({"error": str(e)}), 500


def run_ocr_on_image(img):
    """
    Run OCR on the part of the image where the meter digits are.
    This uses a simple center crop – adjust ratios for your setup!
    """
    h, w, _ = img.shape

    # ---- ROI: adjust these based on how your counter looks ----
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    y1 = int(h * 0.30)
    y2 = int(h * 0.80)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # If your digits are white on black, invert:
    # thresh = cv2.bitwise_not(thresh)

    # OCR with digits only
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    raw = text.strip()
    digits = "".join(c for c in raw if c.isdigit())

    print("OCR raw:", repr(raw), "digits:", digits)
    return digits or None


def append_reading(timestamp, reading, ip):
    file_exists = os.path.exists(READINGS_CSV)
    with open(READINGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "reading", "ip"])
        writer.writerow([timestamp, reading, ip])


def append_image_log(timestamp, width, height, size_bytes, ip):
    file_exists = os.path.exists(IMAGE_LOG_CSV)
    with open(IMAGE_LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "width", "height", "size_bytes", "ip"])
        writer.writerow([timestamp, width, height, size_bytes, ip])


@app.route("/last")
def last():
    """Show the last uploaded image."""
    if not os.path.exists(LAST_IMG_PATH):
        return "❌ No image received yet", 404
    return send_file(LAST_IMG_PATH, mimetype="image/jpeg")


@app.route("/log/readings")
def log_readings():
    """Download the readings CSV (optional helper)."""
    if not os.path.exists(READINGS_CSV):
        return "❌ No readings yet", 404
    return send_file(
        READINGS_CSV,
        mimetype="text/csv",
        as_attachment=True,
        download_name="readings.csv",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

