#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
from wordcloud import STOPWORDS
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')


# In[2]:


df = pd.read_csv('df_clean.csv')


# In[5]:


tweet_text = df['text']

lower_full_text = tweet_text.str.lower().str.cat(sep=' ') # concatenate all strings in the series into one string
word_tokens = word_tokenize(lower_full_text)
tokens = list()
my_stop_words = ['nft','will','de','m','o','news','s','t',"https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'] + list(STOPWORDS)
for word in word_tokens:
    if word.isalpha() and word not in my_stop_words:
        tokens.append(word)
token_dist = FreqDist(tokens)
dist = pd.DataFrame(token_dist.most_common(20),columns=['Word', 'Frequency'])
#dist.plot.bar(x='Word',y='Frequency')


# In[23]:


porter = PorterStemmer()
stemmed_tokens =[porter.stem(word) for word in tokens]
stemmed_token_dist = FreqDist(stemmed_tokens)
stemmed_dist = pd.DataFrame(stemmed_token_dist.most_common(20),columns=['Stemmed Word', 'Frequency'])
fig = plt.figure(figsize=(15, 10))
ax = stemmed_dist.plot.bar(x='Stemmed Word', y='Frequency')
ax.set_title('Top 20 Popular Words')
plt.show()


# In[ ]:




