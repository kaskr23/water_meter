import os
from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Cloud OCR server online", 200

@app.route("/upload", methods=["POST"])
def upload():
    try:
        img_bytes = request.data
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "invalid image"}), 400

        cv2.imwrite("/tmp/last_upload.jpg", img)
        h, w, _ = img.shape
        print(f"📸 Received image {w}x{h}, {len(img_bytes)} bytes")

        return jsonify({"status": "ok", "size": len(img_bytes)}), 200
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
