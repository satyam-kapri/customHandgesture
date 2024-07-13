from flask import Flask, render_template, Response,request

from helpers import generate_frames,capture_training_data,set_label
from flask_cors import CORS



app = Flask(__name__)    

CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/detectgesture')
def detectgesture():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capturegesture')
def capturegesture():
    return Response(capture_training_data(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/setlabel',methods=['POST'])
def setlabel():
    label = request.form['label']
    print("label:",label)
    set_label(label)
    return "success"

if __name__ == "__main__":
    app.run(debug=True)
