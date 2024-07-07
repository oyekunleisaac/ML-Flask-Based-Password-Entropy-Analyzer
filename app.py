import numpy as np
from flask import Flask, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SpatialDropout1D, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import json
from keras.models import load_model

# Custom SpatialDropout1D to handle deserialization issues
class CustomSpatialDropout1D(SpatialDropout1D):
    def __init__(self, rate, **kwargs):
        kwargs.pop('noise_shape', None)
        kwargs.pop('seed', None)
        kwargs.pop('trainable', None)
        super(CustomSpatialDropout1D, self).__init__(rate, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        return cls(**config)

# Custom LSTM to handle deserialization issues
class CustomLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs.pop('time_major', None)
        super(CustomLSTM, self).__init__(units, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)
        return cls(**config)

# Register custom layers
tf.keras.utils.get_custom_objects().update({'SpatialDropout1D': CustomSpatialDropout1D, 'LSTM': CustomLSTM})

app = Flask(__name__)

# Load the pre-trained model with custom objects
lstm_model = load_model('models/lstm_model.h5', custom_objects={'SpatialDropout1D': CustomSpatialDropout1D, 'LSTM': CustomLSTM})

# Load tokenizer configuration
with open('models/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = tokenizer_data.get('word_index')

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
