# Description: This file contains the code for the sentiment visualization page.


# importing needed libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

st.sidebar.header("Select tag")
st.sidebar.write('Our TOP 5 tags are: covid, news, technology, food, sports.')
tag_name = st.sidebar.text_input('Enter the tag name from our TOP5 list:','covid')


# import os

# # Get the current working directory
# current_dir = os.getcwd()
# # Print the list of files in the current directory
# st.write(os.listdir(current_dir))
# # Build the file path
# file_path = os.path.join(current_dir, f'pages/{tag_name}_clean.csv')

# st.write(file_path)


# get data
# In prod env use this.
df = pd.read_csv(f'frontend_streamlit/pages/{tag_name}_clean.csv')
# In dev env use this.
# df = pd.read_csv(f'pages/{tag_name}_clean.csv')
tweet_text = df['text']
sid = SentimentIntensityAnalyzer()

df['compound'] = tweet_text.apply(sid.polarity_scores)
extract_values = lambda x: pd.Series([x['neg'], x['neu'], x['pos'], x['compound']], 
                                     index=['neg', 'neu', 'pos', 'compound'])

# apply lambda function to create new columns
df[['neg', 'neu', 'pos', 'compound']] = df['compound'].apply(extract_values)
df.dropna(subset=['compound'],inplace=True)

# Define date range filter
start_date_str = '2020-01-01 00:00:00'
start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').date()
end_date_str = '2023-01-01 23:59:59'
end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').date()
start_date, end_date = st.sidebar.date_input('Date range:', value=(start_date, end_date))

filtered_data = df[(df['timestamp'] >= start_date_str) & (df['timestamp'] <= end_date_str)]

# Convert timestamp column to datetime type
filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])

# Group the data by month and calculate the mean compound score
grouped_data = filtered_data.groupby(pd.Grouper(key='timestamp', freq='M')).mean(numeric_only=True)['compound']
# Create a line chart changes over time
fig = px.line(grouped_data, x=grouped_data.index, y='compound', title='Average Sentiment Score over time')
fig.update_layout(margin=dict(l=80, r=20, t=40, b=20),
    width=800,
    height=400,
    xaxis_title="Month",
    xaxis_range=[start_date, end_date])
st.title("Sentiment Trend by Month")
st.plotly_chart(fig)


def visualise_sentiments(data):
    scores = []
    for sentence in data:
        # Compute sentiment scores for each word in the sentence
        word_scores = [sid.polarity_scores(word)["compound"] for word in sentence.split()]
        # Compute the average sentiment score for the sentence
        sentence_score = np.mean(word_scores)
        scores.append(word_scores + [sentence_score])
    
    # Create DataFrame to store sentiment scores for each sentence
    df = pd.DataFrame(scores, columns=[word for word in data[0].split()] + ["Sentence Score"])
    df = df.reset_index(drop=True).loc[:,~df.columns.duplicated()]

    # Create heatmap visualization using streamlit
    st.dataframe(df.style.background_gradient(cmap='PiYG', axis=None))
    st.write("The darker the color, the more negative the sentiment.")
    fig = px.imshow(df, color_continuous_scale='PiYG', title='Sentiment Heatmap')
    st.plotly_chart(fig)

st.sidebar.header("Select tweet to check sentiment score")
text_select = st.sidebar.selectbox("Choose a tweet", df['text'])

st.title("Sentiment Analysis")
st.text("Tweet Text: {}".format(text_select))
visualise_sentiments([text_select])

