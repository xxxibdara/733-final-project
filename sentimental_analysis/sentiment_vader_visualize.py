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


sid.polarity_scores(tweet_text[10])


# In[5]:


sns.set(rc={'figure.figsize':(30,1)})

def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")


# In[6]:


visualise_sentiments({
    "Sentence":["SENTENCE"] + tweet_text[10].split(),
    "Sentiment":[sid.polarity_scores(tweet_text[10])["compound"]] + [sid.polarity_scores(word)["compound"] for word in tweet_text[10].split()]
})


# In[ ]:




