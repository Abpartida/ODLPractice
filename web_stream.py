from flask import Flask, Response
import cv2

app = Flask(__name__)

# Adjust this path or method to get your latest processed frame
def generate_frames():
    cap = cv2.VideoCapture(0)  # Change this to your OAK-D frame grabber if needed
    while True:
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Live Stream</h1><img src="/video"/>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)