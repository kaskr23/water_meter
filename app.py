from flask import Flask, request, jsonify
import cv2, numpy as np, pytesseract

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Cloud OCR server online", 200

@app.route("/upload", methods=["POST"])
def upload():
    img_bytes = request.data
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    digits = "".join(c for c in text if c.isdigit())
    return jsonify({"raw": text.strip(), "digits": digits}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
