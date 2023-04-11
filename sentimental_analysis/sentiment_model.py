#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing needed libraries
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# NLTK tools for text processing
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df = pd.read_csv('df_clean.csv')


# In[3]:


tweet_text = df['text']
sid = SentimentIntensityAnalyzer()


# In[4]:


df['compound'] = tweet_text.apply(sid.polarity_scores)


# In[5]:


extract_values = lambda x: pd.Series([x['neg'], x['neu'], x['pos'], x['compound']], 
                                     index=['neg', 'neu', 'pos', 'compound'])

# apply lambda function to create new columns
df[['neg', 'neu', 'pos', 'compound']] = df['compound'].apply(extract_values)
df.dropna(subset=['compound'], inplace=True)


# In[9]:


def detect_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
df['sentiment'] = tweet_text.apply(detect_sentiment)


# In[11]:


df['num_comments'] = df['num_comments'].astype(float)
df['num_retweets'] = df['num_retweets'].astype(float)
df['num_views'] = df['num_views'].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'])


# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

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


# In[22]:


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


# In[ ]:




