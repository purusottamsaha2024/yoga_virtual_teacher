from flask import Flask, render_template, Response
import cv2
from detection import pose_detection

app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(pose_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
