# Description: This file contains the code for the sentiment model

# importing needed libraries
import pandas as pd
from textblob import TextBlob
import streamlit as st

# NLTK tools for text processing
import re, nltk
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

tag = 'technology'
df = pd.read_csv(f'{tag}_clean.csv')

tweet_text = df['text']
sid = SentimentIntensityAnalyzer()

df['compound'] = tweet_text.apply(sid.polarity_scores)

extract_values = lambda x: pd.Series([x['neg'], x['neu'], x['pos'], x['compound']], 
                                     index=['neg', 'neu', 'pos', 'compound'])

# apply lambda function to create new columns
df[['neg', 'neu', 'pos', 'compound']] = df['compound'].apply(extract_values)
df.dropna(subset=['compound'], inplace=True)

def detect_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
df['sentiment'] = tweet_text.apply(detect_sentiment)

df['text_tokens'] = df['text'].apply(word_tokenize)
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(df['text_tokens'].apply(lambda x: ' '.join(x)))

# Combine the text features with the other features
X = pd.concat([pd.DataFrame(X_text.toarray()), df[['num_comments', 'num_retweets', 'num_views', 'sentiment']]], axis=1)
y = df['compound']

# Remove rows with missing values
X = X.dropna()
y = y.iloc[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=30)

# Create a Random Forest model and fit it to the training data
rf_model = RandomForestRegressor(n_estimators=150, random_state=30)
rf_model.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


# test our model
text_input = "In today's Tech Friend: Everyone can improve online security with these password tips. But the whole password system is dumb, insecure and sets us up to fail. Our long term mission: Kill passwords forever. It's possible!."
num_comments = 100
num_retweets = 500
num_views = 50
sentiment = detect_sentiment(text_input)

# Create a dataframe with the input text and feature values
input_df = pd.DataFrame([[text_input, num_comments, num_retweets, num_views, sentiment]], columns=['text', 'num_comments', 'num_retweets', 'num_views', 'sentiment'])

input_df['text_tokens'] = input_df['text'].apply(word_tokenize)
X_text_input = vectorizer.transform(input_df['text_tokens'].apply(lambda x: ' '.join(x)))
X_input = pd.concat([pd.DataFrame(X_text_input.toarray()), input_df[['num_comments', 'num_retweets', 'num_views', 'sentiment']]], axis=1)
y_input_pred = rf_model.predict(X_input)

print("Predicted compound score:", y_input_pred)





