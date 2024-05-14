from flask import Flask, render_template, jsonify,send_file
import os

app = Flask(__name__)

# Path to the file with image paths and reports
DATA_FILE_PATH = "/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/frontend/test_file"
current_line_index = 0

# Read the data file into a list of dictionaries
def read_data_file():
    data = []
    with open(DATA_FILE_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split(', ', 1)
            if len(parts) == 2:
                image_path, report = parts
                data.append({'image_path': image_path, 'report': report})
    return data

data = read_data_file()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/restart', methods=['POST'])
def restart():
    global current_line_index
    current_line_index = 0
    return jsonify(success=True)

@app.route('/next_image', methods=['GET'])
def next_image():
    global current_line_index
    if current_line_index < len(data):
        result = data[current_line_index]
        current_line_index += 1
    else:
        result = {'image_path': '', 'report': 'No more data available.'}
    return jsonify(result)

@app.route('/static<path:path>')
def send_report_file(path):
    return send_file(path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)