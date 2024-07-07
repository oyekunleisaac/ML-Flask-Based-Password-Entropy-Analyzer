# Password Analysis Flask App

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python run.py`
4. Open your browser and go to `http://127.0.0.1:5000/`

## Models

- Random Forest Classifier and LSTM Neural Network are used for password complexity analysis.

## Usage

- Enter a password on the main page to analyze its complexity.
- The app will return the predictions from both models.

### Project Description: Password Entropy Analyzer

Overview:
The Password Entropy Analyzer is a web-based application designed to evaluate the strength of user passwords. By analyzing passwords, typing behavior, and context, the system provides comprehensive feedback to help users create more secure passwords.

What the System Does:
1. Password Analysis: Users enter a password, and the system evaluates its strength.
2. Behavioral Analysis: The system measures how quickly the user types the password and provides feedback based on typing speed.
3. Context Awareness: Users select the context in which the password will be used (e.g., banking, email), and the system offers specific advice based on the selected context.

How It Works:
1. User Input: The user enters a password and selects a context (general, banking, or email). The typing time is also recorded.
2. Backend Processing: The entered password is sent to a backend server built using Flask (a Python web framework).
3. Model Analysis: The backend server uses a pre-trained LSTM (Long Short-Term Memory) neural network model to analyze the password. This model was trained on a dataset of passwords to predict their strength.
4. Feedback Generation: The server evaluates the password's strength, provides feedback based on the typing speed, and gives context-specific advice.
5. Display Results: The analyzed results are displayed on the same web page, showing the password's strength, a prediction score, behavioral analysis, and context-specific warnings.

Models Used:
1. LSTM Model: An advanced neural network model that excels at analyzing sequences, such as text. This model was trained on a large dataset of passwords to learn patterns that indicate password strength. The training process involved:
   - Data Collection: Gathering a diverse set of passwords.
   - Model Training: Using TensorFlow, a popular machine learning library, to train the LSTM model. The training process involved feeding the password data into the model and adjusting its parameters to minimize prediction errors.
   - Custom Layers: Custom implementations of LSTM and SpatialDropout1D layers to handle specific configurations during the model loading phase.

User Interface:
The user interacts with a clean, simple web interface. They enter their password, select the context, and submit the form. The system then displays the results directly on the page, providing immediate feedback on their password's strength and personalized advice.

This project aims to help users create stronger, more secure passwords by providing a detailed analysis that goes beyond simple strength indicators, considering user behavior and the specific use case of the password.

