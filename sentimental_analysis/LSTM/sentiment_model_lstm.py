from keras.models import load_model
import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
import numpy as np
# from keras.preprocessing.text import Tokenizer
import pickle


# Load the fitted encoder object
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(input_string, model_path, tokenizer, max_seq_length=250):
    # Load the saved model
    model = load_model(model_path)

    # Preprocess the input string
    input_string = [input_string]  # Convert the string to a list containing the string
    tokenized_input = tokenizer.texts_to_sequences(input_string)
    padded_input = tf.keras.utils.pad_sequences(tokenized_input, maxlen=max_seq_length)

    # Predict the sentiment
    prediction = model.predict(padded_input)

    # Decode the prediction
    sentiment_class = encoder.inverse_transform([np.argmax(prediction)])

    return sentiment_class[0]

# Example usage:
input_string = "A story which continue today. Received a call from my patient that I took care in Covid ward,HKL last year, makcik came to see me again today to give kuih raya made by her & husband. Kuih Semperit, Tart Nenas & Cornflakes. Ya Allah, makcik, you deserve all love in this world"
model_path = "Bidirectional_LSTM.h5"
sentiment = predict_sentiment(input_string, model_path, tokenizer)
print("The sentiment of the input string is:", sentiment)

def get_sentiment(text):
    model_path = "Bidirectional_LSTM.h5"
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    sentiment = predict_sentiment(input_string, model_path, tokenizer)
    return sentiment