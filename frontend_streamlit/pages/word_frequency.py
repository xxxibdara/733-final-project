# Description: This file contains the code for the word frequency page.

# importing needed libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# NLTK tools for text processing
import re, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from nltk import FreqDist
from wordcloud import STOPWORDS
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')


st.sidebar.header("Select tag")
st.sidebar.write('Our TOP 5 tags are: covid, news, technology, food, sports.')

tag_name = st.sidebar.text_input('Enter the tag name from our TOP5 list:','covid')

# In prod env use this.
df = pd.read_csv(f'frontend_streamlit/pages/{tag_name}_clean.csv')

# In dev env use this.
# df = pd.read_csv(f'pages/{tag_name}_clean.csv')
text_df = df['text']

@st.cache_data
def word_frequency(tweet_text):
    lower_full_text = tweet_text.str.lower().str.cat(sep=' ') # concatenate all strings in the series into one string
    word_tokens = word_tokenize(lower_full_text)
    tokens = list()
    my_stop_words = ['nft','will','de','m','o','news','s','t',"https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'] + list(STOPWORDS)
    for word in word_tokens:
        if word.isalpha() and word not in my_stop_words:
            tokens.append(word)

    porter = PorterStemmer()
    stemmed_tokens =[porter.stem(word) for word in tokens]
    stemmed_token_dist = FreqDist(stemmed_tokens)
    stemmed_dist = pd.DataFrame(stemmed_token_dist.most_common(20),columns=['Word', 'Frequency'])

    return stemmed_dist

stemmed_dist = word_frequency(text_df)
# create the plot for the top20 most common words using streamlit
st.title("Top 20 most common words")
st.text("The top 20 most common words in the tag")

st.bar_chart(stemmed_dist.set_index('Word')['Frequency'], use_container_width=True)

st.text("The top 20 most common words are:")
st.write(stemmed_dist)


# create the word cloud using streamlit
comments = []
stop_words = set(stopwords.words('english')) | set(['covid','u','s','t',"https", "co", "RT", 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'rt', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'])

for words in text_df:
    only_letters = re.sub("[^a-zA-Z]", " ",words)
    tokens = nltk.word_tokenize(only_letters) #tokenize the sentences
    lower_case = [l.lower() for l in tokens] #convert all letters to lower case
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) #Remove stopwords from the comments
    comments.append(' '.join(filtered_result))


unique_string=(" ").join(comments)
wordcloud = WordCloud(width = 2000, height = 1000,background_color='black').generate(unique_string)

# Display the word cloud
fig, ax = plt.subplots(figsize=(20, 12))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")

st.title("Word Cloud")
st.text(f"Word Cloud for #{tag_name} tag")
st.pyplot(fig)
