import pandas as pd
import re
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

def clean_data(input_text):
    stop_words_list = pd.read_csv("models/data_dir/stopwordbahasa.csv")
    stop_words_list = stop_words_list.to_numpy()

    # Menghilangkan karakter hasil scrapping yang kurang baik (unnecessary character)

    input_text = re.sub(r"[^A-Za-z0-9\s]+", "", input_text)
    input_text = re.sub("\n"," ", input_text)
    input_text = [word.lower() for word in input_text.split() if word.lower() not in stop_words_list]

    return input_text

def tokenize(input_text):
    input_text = [input_text]

    with open('models/model_dir/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    testing_sequences = tokenizer.texts_to_sequences(input_text)
    padded_sequence = pad_sequences(testing_sequences,maxlen=150, padding='post')
    
    return padded_sequence

def predict_text(input_text):
    clean_text = clean_data(input_text)
    tokenized_text = tokenize(clean_text)
    
    model = tf.keras.models.load_model("models/model_dir/best_model.h5")
    result = model.predict(tokenized_text)
    result_class = "Valid" if result > 0.5 else "Hoaks"
    
    return result_class