import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "model.pt"
LAST_IMAGE_PATH = "/tmp/last.jpg"
LAST_ANNOTATED_PATH = "/tmp/last_annotated.jpg"
# ----------------------------------------

app = Flask(__name__)

# Load YOLO model globally
yolo_model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded")

last_reading = None

def get_reading_from_image(img_bgr, conf_threshold=0.3, save_annotated_path=None):
    """
    img_bgr: BGR numpy array
    returns: reading string like '012345'
    optionally saves an annotated image if save_annotated_path is given.
    """
    results = yolo_model.predict(img_bgr, conf=conf_threshold, verbose=False)[0]
    boxes_data = results.boxes.data.cpu().numpy().tolist()  # [x1,y1,x2,y2,score,class_id]

    boxes = []
    for x1, y1, x2, y2, score, cls_id in boxes_data:
        boxes.append([float(x1), float(y1), float(x2), float(y2), int(cls_id), float(score)])

    if not boxes:
        return ""

    # Sort left → right
    boxes.sort(key=lambda b: b[0])

    digits = []
    annotated = img_bgr.copy()

    for x1, y1, x2, y2, cls_id, score in boxes:
        digit = str(cls_id)  # class_id == digit
        digits.append(digit)

        # draw boxes if debug image requested
        if save_annotated_path is not None:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 0, 255), 2)
            cv2.putText(
                annotated,
                digit,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    reading = "".join(digits)

    if save_annotated_path is not None:
        cv2.imwrite(save_annotated_path, annotated)

    return reading

@app.route("/")
def index():
    return f"✅ YOLO water-meter server online. Last reading: {last_reading}"

@app.route("/upload", methods=["POST"])
def upload():
    global last_reading

    try:
        data = request.data
        if not data:
            return jsonify({"error": "no image data"}), 400

        # Decode JPEG bytes to BGR image
        file_bytes = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "failed to decode JPEG"}), 400

        # Save last image for debug
        cv2.imwrite(LAST_IMAGE_PATH, img)

        # Run YOLO-based meter reading
        reading = get_reading_from_image(img, conf_threshold=0.3, save_annotated_path=LAST_ANNOTATED_PATH)
        last_reading = reading

        return jsonify({
            "status": "ok",
            "reading": reading
        })

    except Exception as e:
        print("❌ Error in /upload:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/last")
def last():
    if not os.path.exists(LAST_IMAGE_PATH):
        return "No image yet", 404
    return send_file(LAST_IMAGE_PATH, mimetype="image/jpeg")

@app.route("/debug/annotated")
def debug_annotated():
    if not os.path.exists(LAST_ANNOTATED_PATH):
        return "No annotated image yet", 404
    return send_file(LAST_ANNOTATED_PATH, mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
