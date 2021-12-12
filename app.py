# Import library
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model
import models.prediction_util as pred

# Create flask app
app = Flask(__name__)

# Render home page
@app.route('/')
def home():
  return render_template('home.html')

# Render predict crop recommendation page
@app.route('/predict-hoax', methods=['POST'])
def hoax_predict():
  text_input = request.form['text_input']

  if len(text_input) >= 50:
    result = pred.predict_text(text_input)
    return render_template('result.html', prediction=result)
  else:
    return render_template('home.html')

# Run flask app
if __name__ == '__main__':
  app.run(debug=True)