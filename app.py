import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
pickle_input = open("Simran_NuthalapatiYesukumar_model.pkl","rb")
random_forest_model = pickle.load(pickle_input)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    input_feat = [x for x in request.form.values()]
    final_model_features = [np.array(input_feat)]
    preds = random_forest_model.predict(final_model_features)
    return render_template('index.html', prediction_text = 'The species that the fish belongs to is {}'.format(str(preds)))

if __name__=='__main__':
    app.run()