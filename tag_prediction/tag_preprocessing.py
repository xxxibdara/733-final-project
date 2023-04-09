import os
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# read dataset from input file path
file_path = input('Enter a file path: ')
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['tags'])
    df['tags'] = df['tags'].str.split('|')
else:
    print('The specified file does NOT exist')

# define stopwords
stop_words = set(stopwords.words("english"))


# tokenize and extract information from raw text
def cleaner(text):
    # extract paragraph from HTML format
    text = BeautifulSoup(text).get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize text and remove stopwords
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return tokens


# apply cleaner function to the dataset
df['article_cleaned'] = df['article'].apply(cleaner)
df['title_cleaned'] = df['title'].apply(cleaner)
df['query'] = df.apply(lambda x: x['title_cleaned'] +
                       x['article_cleaned'], axis=1)

# save cleaned dataset
df_train = df[['query', 'tags']]
df_train.to_csv('train_clean.csv', index=False)
