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


# In[2]:


covid = pd.read_csv('covid.csv')
covid


# In[3]:


def data_cleaning(df):
    df['user'] = df['user'].replace('[^a-zA-Z0-9]', '', regex=True)
    df['text'] = df['text'].replace('#', '', regex=True)
    df.dropna(subset=['text'], inplace=True)#3515
    df.drop_duplicates(subset=['text'], inplace=True)#3239
    return df

data_cleaning(covid)


# In[4]:


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

covid['num_comments'] = covid['num_comments'].apply(convert_value)
covid['num_retweets'] = covid['num_retweets'].apply(convert_value)
covid['num_views'] = covid['num_views'].apply(convert_value)


# In[5]:


def convert_timestamp(df):
    df = df[df["timestamp"] != "0"]  # Remove rows where timestamp value is "0"
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"))
    return df

covid = convert_timestamp(covid) #3238


# In[ ]:





# In[6]:


covid


# In[ ]:





# In[7]:


tweet_text = covid['text']
tweet_text


# In[8]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[9]:


# Transforming the reviews data by removing stopwords, using regular expressions module to accept only letters,
# tokenizing those words and then making all the words lower case for consistency.
from wordcloud import STOPWORDS
comments = []
stop_words = set(stopwords.words('english')) | set(['s','t',"https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'])

for words in tweet_text:
   only_letters = re.sub("[^a-zA-Z]", " ",words)
   tokens = nltk.word_tokenize(only_letters) #tokenize the sentences
   lower_case = [l.lower() for l in tokens] #convert all letters to lower case
   filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) #Remove stopwords from the comments
   comments.append(' '.join(filtered_result))


# In[10]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
unique_string=(" ").join(comments)
wordcloud = WordCloud(width = 2000, height = 1000,background_color='black').generate(unique_string)
plt.figure(figsize=(20,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[11]:


import nltk
nltk.download('punkt')
  
from nltk.tokenize import word_tokenize
from nltk import FreqDist

lower_full_text = tweet_text.str.lower().str.cat(sep=' ') # concatenate all strings in the series into one string
word_tokens = word_tokenize(lower_full_text)
tokens = list()
my_stop_words = ['nft','will','de','m','o','news','s','t',"https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'] + list(STOPWORDS)
for word in word_tokens:
    if word.isalpha() and word not in my_stop_words:
        tokens.append(word)
token_dist = FreqDist(tokens)
dist = pd.DataFrame(token_dist.most_common(20),columns=['Word', 'Frequency'])
dist.plot.bar(x='Word',y='Frequency')


# In[12]:


from nltk.stem import PorterStemmer

porter = PorterStemmer()
stemmed_tokens =[porter.stem(word) for word in tokens]
stemmed_token_dist = FreqDist(stemmed_tokens)
stemmed_dist = pd.DataFrame(stemmed_token_dist.most_common(20),columns=['Stemmed Word', 'Frequency'])
stemmed_dist.plot.bar(x='Stemmed Word',y='Frequency')


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS

#my_stop_words = ["https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'] + list(STOPWORDS)
vect = CountVectorizer(stop_words=my_stop_words, ngram_range=(2,2))
bigrams = vect.fit_transform(tweet_text)
bigram_df = pd.DataFrame(bigrams.toarray(), columns=vect.get_feature_names())
bigram_frequency = pd.DataFrame(bigram_df.sum(axis=0)).reset_index()
bigram_frequency.columns = ['bigram', 'frequency']
bigram_frequency = bigram_frequency.sort_values(by='frequency', ascending=False).head(20)


# In[14]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[15]:


sid.polarity_scores(tweet_text[10])


# In[16]:


import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(30,1)})

def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")


# In[17]:


visualise_sentiments({
    "Sentence":["SENTENCE"] + tweet_text[10].split(),
    "Sentiment":[sid.polarity_scores(tweet_text[10])["compound"]] + [sid.polarity_scores(word)["compound"] for word in tweet_text[10].split()]
})


# In[18]:


# create a new DataFrame column for compound 
covid['compound'] = tweet_text.apply(sid.polarity_scores)
covid


# In[19]:


#split compound column
# define lambda function to extract values
extract_values = lambda x: pd.Series([x['neg'], x['neu'], x['pos'], x['compound']], 
                                     index=['neg', 'neu', 'pos', 'compound'])

# apply lambda function to create new columns
covid[['neg', 'neu', 'pos', 'compound']] = covid['compound'].apply(extract_values)


# In[20]:


import nltk
nltk.download('punkt')
from textblob import TextBlob


# In[21]:


# define a function that accepts text and returns the polarity
def detect_sentiment(text):
    
    #Converts the text into textblob object and then retuns
    #the polarity.
    blob = TextBlob(text)
    
    # return the polarity
    return blob.sentiment.polarity
    #return blob.sentiment.subjectivity


# In[22]:


covid['sentiment'] = tweet_text.apply(detect_sentiment)


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(covid.compound, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity')
plt.show()


# In[24]:


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
    plt.ylabel('Mean Compound Score')
    plt.show()
    
start_time = '2020-01-01 00:00:00'
end_time = '2021-01-01 23:59:59'
plot_sentiment_trend(covid, start_time, end_time)


# In[25]:


df_model = covid.copy()


# In[26]:


df_model['num_comments'] = df_model['num_comments'].astype(float)
df_model['num_retweets'] = df_model['num_retweets'].astype(float)
df_model['num_views'] = df_model['num_views'].astype(float)
df_model['timestamp'] = pd.to_datetime(df_model['timestamp'])


# In[27]:


df_train = df_model.iloc[:, :3000]
df_test = df_model.iloc[3001:]
df_test = df_test[['text','num_comments','num_retweets','num_views']]


# Decision tree with multiple variables

# In[28]:


#Define X as a DataFrame with multiple columns
X_dt = df_model[['text', 'num_comments','num_retweets','num_views']]
y_dt = df_model.compound 

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Define a function that accepts a vectorizer and a decision tree regressor, and returns the mean squared error, accuracy, precision, recall, and F1 score
def predict_sentiment_score(vectorizer, regressor):
    
    # create document-term matrices using the vectorizer
    X_train_text = vectorizer.fit_transform(X_train['text'])
    X_test_text = vectorizer.transform(X_test['text'])
    
    # combine the text, likes, and sentiment data into a sparse matrix
    import scipy.sparse as sp
    X_train_comm = sp.csr_matrix(X_train['num_comments'].values.reshape(-1, 1))
    X_test_comm = sp.csr_matrix(X_test['num_comments'].values.reshape(-1, 1))
    X_train_re = sp.csr_matrix(X_train['num_retweets'].values.reshape(-1, 1))
    X_test_re = sp.csr_matrix(X_test['num_retweets'].values.reshape(-1, 1))
    X_train_views = sp.csr_matrix(X_train['num_views'].values.reshape(-1, 1))
    X_test_views = sp.csr_matrix(X_test['num_views'].values.reshape(-1, 1))
    X_train_combined = sp.hstack([X_train_text,X_train_comm, X_train_re, X_train_views])
    X_test_combined = sp.hstack([X_test_text,X_test_comm, X_test_re, X_test_views])
    
    # train the decision tree regressor
    regressor.fit(X_train_combined, y_train)
    
    # make predictions on the test set
    y_pred = regressor.predict(X_test_combined)
    
    # calculate and return the mean squared error, accuracy, precision, recall, and F1 score
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test > 0, y_pred > 0)
    prec = precision_score(y_test > 0, y_pred > 0)
    rec = recall_score(y_test > 0, y_pred > 0)
    f1 = f1_score(y_test > 0, y_pred > 0)
    
    return mse, acc, prec, rec, f1

# create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# create a decision tree regressor
dt_regressor = DecisionTreeRegressor()

# test the performance of the decision tree regressor with TF-IDF vectorization
tfidf_mse, tfidf_acc, tfidf_prec, tfidf_rec, tfidf_f1 = predict_sentiment_score(tfidf_vectorizer, dt_regressor)

print(f"Mean Squared Error: {tfidf_mse}") 
print(f"Accuracy: {tfidf_acc}")
print(f"Precision: {tfidf_prec}")
print(f"Recall: {tfidf_rec}")
print(f"F1: {tfidf_f1}")


# In[33]:


def predict_sentiment_score(vectorizer, regressor, data):
    
    # create document-term matrices using the vectorizer
    X_text = vectorizer.transform(data['text'])
    
    # combine the text, likes, and sentiment data into a sparse matrix
    import scipy.sparse as sp
    X_comm = sp.csr_matrix(data['num_comments'].values.reshape(-1, 1))
    X_re = sp.csr_matrix(data['num_retweets'].values.reshape(-1, 1))
    X_views = sp.csr_matrix(data['num_views'].values.reshape(-1, 1))
    X_combined = sp.hstack([X_text, X_comm, X_re, X_views])
    
    # make predictions on the new data
    y_pred = regressor.predict(X_combined)
    
    return y_pred

# use the trained model to predict compound values for the new data
y_pred = predict_sentiment_score(tfidf_vectorizer, dt_regressor, df_test)

df_test['predicted_compound'] = y_pred


# In[90]:


#random forest
df_rf = covid.iloc[:, :3000]


# In[115]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Preprocess the text data
# df_rf['year'] = pd.to_datetime(df_rf['timestamp']).dt.year
# df_rf['month'] = pd.to_datetime(df_rf['timestamp']).dt.month
# df_rf['day'] = pd.to_datetime(df_rf['timestamp']).dt.day
# df_rf['hour'] = pd.to_datetime(df_rf['timestamp']).dt.hour
df_rf['text_tokens'] = df_rf['text'].apply(word_tokenize)
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(df_rf['text_tokens'].apply(lambda x: ' '.join(x)))

# Combine the text features with the other features
X_rf = pd.concat([pd.DataFrame(X_text.toarray()), df_rf[['num_comments', 'num_retweets', 'num_views', 'sentiment']]], axis=1)
y_rf = df_rf['compound']

# Remove rows with missing values
X_rf = X_rf.dropna()
y_rf = y_rf.iloc[X_rf.index]

# Split the data into training and testing sets
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.5, random_state=30)

# Create a Random Forest model and fit it to the training data
rf_model = RandomForestRegressor(n_estimators=150, random_state=30)
rf_model.fit(X_rf_train, y_rf_train)

# Use the model to make predictions on the test data
y_rf_pred = rf_model.predict(X_rf_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_rf_test, y_rf_pred)
r2 = r2_score(y_rf_test, y_rf_pred)
print("MSE:", mse)
print("R-squared:", r2)


# In[118]:


# Preprocess the text data
text_input = "According to new research people hospitalized with #COVID19 this past flu season were more to die than people hospitalized with influenza, especially if they were unvaccinated against the coronavirus."
# Add other features
num_comments = 100
num_retweets = 500
num_views = 50
sentiment = detect_sentiment(text_input)

# Create a dataframe with the input text and feature values
input_df = pd.DataFrame([[text_input, num_comments, num_retweets, num_views, sentiment, year, month, day, hour]], columns=['text', 'num_comments', 'num_retweets', 'num_views', 'sentiment', 'year', 'month', 'day', 'hour'])

# Preprocess the text data
input_df['text_tokens'] = input_df['text'].apply(word_tokenize)
X_text_input = vectorizer.transform(input_df['text_tokens'].apply(lambda x: ' '.join(x)))

# Combine the text features with the other features
X_input = pd.concat([pd.DataFrame(X_text_input.toarray()), input_df[['num_comments', 'num_retweets', 'num_views', 'sentiment']]], axis=1)

# Use the trained model to make predictions on the input data
y_input_pred = rf_model.predict(X_input)

print("Predicted compound score:", y_input_pred)


# In[76]:


covid.tail(15)


# In[ ]:





# In[ ]:





# In[ ]:




