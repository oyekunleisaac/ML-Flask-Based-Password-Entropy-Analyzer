from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.layers import SpatialDropout1D
import json
import time

app = Flask(__name__)

# Custom SpatialDropout1D to handle 'trainable' and 'noise_shape' arguments
class CustomSpatialDropout1D(SpatialDropout1D):
    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        config.pop('noise_shape', None)
        return cls(**config)

# Custom LSTM to handle 'time_major' argument
from tensorflow.keras.layers import LSTM

class CustomLSTM(LSTM):
    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)
        return cls(**config)

# Paths to model files
lstm_model_path = 'models/lstm_model.h5'
tokenizer_path = 'models/tokenizer.json'
max_length_path = 'models/max_length.txt'

# Load the tokenizer
with open(tokenizer_path, 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

# Load the max_length
with open(max_length_path, 'r') as f:
    max_length = int(f.read())

# Load the LSTM model with custom objects
lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects={'SpatialDropout1D': CustomSpatialDropout1D, 'LSTM': CustomLSTM})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_password():
    data = request.json
    password = data.get('password')
    typing_time = data.get('typing_time')
    context = data.get('context')
    
    sequence = tokenizer.texts_to_sequences([password])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length)
    prediction = lstm_model.predict(padded_sequence)[0][0]

    # Categorize the password strength
    if prediction < 0.3:
        strength = 'Weak'
    elif prediction < 0.7:
        strength = 'Medium'
    else:
        strength = 'Strong'

    # Adding behavioral analysis
    if typing_time > 5:
        behavior_analysis = 'You took a while to type the password. Consider a password manager for stronger, more complex passwords.'
    else:
        behavior_analysis = 'You typed the password quickly. Ensure it’s not a common or reused password.'

    # Context awareness
    if context == 'banking':
        context_warning = 'For banking, use a unique password that is not used elsewhere.'
    elif context == 'email':
        context_warning = 'Ensure your email password is strong as it can be a target for hackers.'
    else:
        context_warning = 'Consider the importance of the service when choosing your password.'

    return jsonify({
        'prediction': float(prediction),
        'strength': strength,
        'message': f'The password strength is categorized as {strength}.',
        'behavior_analysis': behavior_analysis,
        'context_warning': context_warning
    })

if __name__ == '__main__':
    app.run(debug=True)
