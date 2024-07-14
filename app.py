from flask import Flask, render_template, Response,request,jsonify,send_from_directory

from helpers import generate_frames,capture_training_data,set_label
from flask_cors import CORS
import os


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


@app.route('/fetchsamples', methods=['POST'])
def fetch_samples():
    sample_dir = 'samples'
    all_samples = []

    # Iterate over each label directory in samples directory
    for label in os.listdir(sample_dir):
        label_dir = os.path.join(sample_dir, label)
        if os.path.isdir(label_dir):
            # Get list of images in current label directory
            images = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
            for image in images:
                # Construct full path to image
                image_path = os.path.join('samples', label, image)
                # Append label and image path to result list
                all_samples.append({"label": label, "image_path": image_path})

    return jsonify({"samples": all_samples})

@app.route('/samples/<path:filename>')
def sample_image(filename):
    return send_from_directory('samples', filename)

if __name__ == "__main__":
    app.run(debug=True)
