from flask import Flask, request, jsonify, make_response
import numpy as np
import base64
import json

app = Flask(__name__)


@app.route('/', methods=["POST"])
def index():
    print("received request for mri processing")
    data = request.json
    base64_mri = data.get("base64_mri", "")
    print(f"length of mri string: {len(base64_mri)}")
    mri = np.frombuffer(base64.b64decode(base64_mri),dtype=float).reshape((172,220,156,1))
    print(f"shape: {mri.shape}")

    # convert mri back to string
    array_str = base64.b64encode(mri.tobytes()).decode('utf-8')
    print(f"len of array while sending it back: {len(array_str)}")
    data = { "base64_mri": base64.b64encode(mri.tobytes()).decode('utf-8') }
    return data

app.run(host="0.0.0.0", port=5050)
