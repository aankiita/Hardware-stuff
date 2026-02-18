from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model("spitting_model.h5")
IMG_SIZE = 64  # match your model

# Open webcam
#cap = cv2.VideoCapture(0)
cap=cv2.VideoCapture("demo\s7.mp4")

# Shared variable for spitting detection
spitting_detected = False

def gen_frames():
    global spitting_detected
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img)[0]
        label = np.argmax(pred)  # softmax
        # label = 1 if pred > 0.5 else 0  # if sigmoid

        if label == 1:
            text = "Spitting"
            color = (0, 0, 255)
            spitting_detected = True
        else:
            text = "Not Spitting"
            color = (0, 255, 0)
            spitting_detected = False

        # Overlay text
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')  # Home page with popup

@app.route('/control')
def control():
    return render_template('control.html')  # Control page

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/spitting_status')
def spitting_status():
    return jsonify({"spitting": spitting_detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
