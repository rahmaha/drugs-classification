import pickle
import numpy as np
from flask import Flask, request, jsonify

dv_path = 'dv.bin'
model_path = 'model_rf.bin'
# load and read file


def load_file(file):
    with open(file, 'rb') as f_in:
        return pickle.load(f_in)


dv = load_file(dv_path)
model = load_file(model_path)


app = Flask("drugs-classification")
@app.route("/predict", methods=['POST'])
def predict():
    patient = request.get_json()
    X_patient = dv.transform(patient)
    y_pred = (model.predict(X_patient))
    get_drug = (y_pred[0])

    result = {
        "drug": get_drug
    }
    return (jsonify(result))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
