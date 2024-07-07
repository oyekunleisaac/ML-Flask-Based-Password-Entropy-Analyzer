import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_models():
    # Load Random Forest model
    with open('models/random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    
    # Load LSTM model
    lstm_model = load_model('models/lstm_model.h5')
    
    # Load tokenizer and max_len from saved states
    with open('models/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    with open('models/max_len.pkl', 'rb') as file:
        max_len = pickle.load(file)
    
    return rf_model, lstm_model, tokenizer, max_len

def predict_password_complexity(password, rf_model, lstm_model, tokenizer, max_len):
    # Tokenize and pad password
    tokenized = tokenizer.texts_to_sequences([password])
    padded = pad_sequences(tokenized, maxlen=max_len)
    
    # Random Forest prediction
    rf_prediction = rf_model.predict(padded)[0]
    
    # LSTM prediction
    lstm_prediction = lstm_model.predict(padded)[0][0]
    
    # Convert predictions to binary classification
    rf_result = int(rf_prediction >= 0.5)
    lstm_result = int(lstm_prediction >= 0.5)
    
    return rf_result, lstm_result
