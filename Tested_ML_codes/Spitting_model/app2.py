from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import requests

app = Flask(__name__)

# ================= CONFIG =================
MODEL_PATH = "spitting_model.h5"
IMG_SIZE = 64
ESP32_IP = "http://10.33.58.150"   # ip

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)

# ================= CAMERA (USB WEBCAM FIX) =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ ERROR: Webcam not opened")

# ================= STATES =================
spitting_detected = False
last_spitting_state = False


# ================= VIDEO GENERATOR =================
def gen_frames():
    global spitting_detected, last_spitting_state

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # ---- ML PREPROCESS (SAFE COPY) ----
        roi = cv2.resize(frame.copy(), (IMG_SIZE, IMG_SIZE))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)

        # ---- Prediction ----
        pred = model.predict(roi, verbose=0)[0]
        label = np.argmax(pred)

        if label == 1:
            spitting_detected = True
            text = "Spitting"
            color = (0, 0, 255)
        else:
            spitting_detected = False
            text = "Not Spitting"
            color = (0, 255, 0)

        # ---- ESP32 ALERT (EDGE TRIGGER) ----
        if spitting_detected and not last_spitting_state:
            try:
                requests.get(f"{ESP32_IP}/auto_alert", timeout=1)
            except:
                pass

        last_spitting_state = spitting_detected

        # ---- Overlay text ----
        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # ---- Encode frame ----
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/control")
def control():
    return render_template("control.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/spitting_status")
def spitting_status():
    return jsonify({"spitting": spitting_detected})


# ================= MAIN =================
if __name__ == "__main__":
    # IMPORTANT: disable reloader to avoid camera glitch
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
