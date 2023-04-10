#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
from wordcloud import STOPWORDS
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


df = pd.read_csv('covid.csv')


# In[4]:


def data_cleaning(df):
    df['user'] = df['user'].replace('[^a-zA-Z0-9]', '', regex=True)
    df['text'] = df['text'].replace('#', '', regex=True)
    df.dropna(subset=['text'], inplace=True)#3515
    df.drop_duplicates(subset=['text'], inplace=True)#3239
    return df

data_cleaning(df)


# In[6]:


def convert_value(value):
    if isinstance(value, int):
        value = str(value)
    value = value.replace(",", "")
    if value.endswith("K"):
        value = value.replace("K", "000").replace(".", "")
    elif value.endswith("M"):
        value = value.replace("M", "000000").replace(".", "")
    value = int(value)
    return value

df['num_comments'] = df['num_comments'].apply(convert_value)
df['num_retweets'] = df['num_retweets'].apply(convert_value)
df['num_views'] = df['num_views'].apply(convert_value)


# In[14]:


def convert_timestamp(df):
    df = df[df["timestamp"] != "0"]  # Remove rows where timestamp value is "0"
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"))
    return df

df = convert_timestamp(df) #3238


# In[16]:


df.to_csv('df_clean.csv')


# In[ ]:





# In[ ]:




