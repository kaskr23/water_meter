import os
from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Cloud OCR server online", 200


@app.route("/upload", methods=["GET", "POST"])
def upload():
    # If you open /upload in a browser (GET) -> just a test message
    if request.method == "GET":
        return "✅ /upload endpoint is ready (send POST with image/jpeg)", 200

    # ESP32 sends POST with image/jpeg body
    try:
        img_bytes = request.data
        if not img_bytes:
            return jsonify({"error": "empty body"}), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "invalid image"}), 400

        # Save last image for debugging
        save_path = "/tmp/last_upload.jpg"
        cv2.imwrite(save_path, img)

        h, w, _ = img.shape
        print(f"📸 Received image {w}x{h}, {len(img_bytes)} bytes")

        return jsonify({"status": "ok", "width": w, "height": h, "size": len(img_bytes)}), 200

    except Exception as e:
        print("❌ Error in /upload:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/last")
def last():
    """Show the last image received, in the browser."""
    path = "/tmp/last_upload.jpg"
    if not os.path.exists(path):
        return "❌ No image received yet", 404

    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
