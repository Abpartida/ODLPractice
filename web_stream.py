from flask import Flask, Response, render_template
import cv2
import time

# IMPORTANT: import the global frame from main
from main import latest_frame

app = Flask(__name__)

def generate_frames():
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # ~20 FPS

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template(
        'live_stream.html',
        title='LYCO TOMI Live Stream',
        stream_url='/video'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
