import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle
import pandas as pd

def load_models():
    file_name = "./FlaskAPI/models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
        features = data['features']
    return model, features

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/predict', methods=['POST'])
def predict():
    # stub input features
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1,17)

    # load model
    model, features = load_models()
    # Align input with features
    input_df = pd.DataFrame([x], columns=features)
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    app.run(debug=True) 
