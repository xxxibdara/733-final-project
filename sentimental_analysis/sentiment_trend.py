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


# In[8]:


def plot_sentiment_trend(data, start_date, end_date):
    # Filter the data by date range
    filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
    # Convert timestamp column to datetime type
    filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
    # Group the data by month and calculate the mean compound score
    grouped_data = filtered_data.groupby(pd.Grouper(key='timestamp', freq='M')).mean()['compound']
    # Plot the data
    plt.figure(figsize=(10,6))
    plt.plot(grouped_data.index, grouped_data.values)
    plt.title('Sentiment Trend by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.show()
    
start_time = '2020-01-01 00:00:00'
end_time = '2021-01-01 23:59:59'
plot_sentiment_trend(df, start_time, end_time)


# In[ ]:





# In[ ]:





# In[ ]:




