from flask import Flask, render_template, request, jsonify
import torch
import os
os.environ["HF_HOME"] = "/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.hf_cache"
import transformers

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)