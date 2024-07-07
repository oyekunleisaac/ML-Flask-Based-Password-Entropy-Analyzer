import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
lstm_model = load_model('models/lstm_model.h5')

# Load tokenizer configuration
with open('models/tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)
    tokenizer = Tokenizer.from_config(tokenizer_config)

# Load max length
with open('models/max_length.txt', 'r') as f:
    max_length = int(f.read().strip())

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    password = data.get('password')
    
    # Preprocess the password to match the training format
    padded_password_array = preprocess_password(password)
    
    # Make the prediction using the model
    prediction = analyze_password(padded_password_array)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

def preprocess_password(password):
    # Tokenize the password using the loaded tokenizer
    sequences = tokenizer.texts_to_sequences([password])
    
    # Pad the sequences to ensure they are the same length as used during training
    padded_password_array = pad_sequences(sequences, maxlen=max_length)
    
    return padded_password_array

def analyze_password(padded_password_array):
    # Use the model to predict the password strength
    prediction = lstm_model.predict(padded_password_array)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
