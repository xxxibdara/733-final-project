# Description: This file contains the code for the sentiment visualization page.


# importing needed libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import sys
# sys.path.append('sentimental_analysis/LSTM')
# import sentiment_model_lstm
import nltk
nltk.download('vader_lexicon')


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf




st.sidebar.title('Top 5 tags')
st.sidebar.write('covid, news, technology, food, sports.')
tag_name = st.sidebar.selectbox('Select a tag:', ['covid', 'news', 'technology', 'food', 'sports'])



# get data
# In prod env use this.
df = pd.read_csv(f'frontend_streamlit/pages/{tag_name}_clean.csv')
# In dev env use this.
# df = pd.read_csv(f'pages/{tag_name}_clean.csv')
tweet_text = df['text'].astype(str)
sid = SentimentIntensityAnalyzer()
# Convert timestamp column to datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])


df['compound'] = tweet_text.apply(sid.polarity_scores)
# df['compound'] = tweet_text.apply(sentiment_model_lstm.get_sentiment)
extract_values = lambda x: pd.Series([x['neg'], x['neu'], x['pos'], x['compound']], 
                                     index=['neg', 'neu', 'pos', 'compound'])

# apply lambda function to create new columns
df[['neg', 'neu', 'pos', 'compound']] = df['compound'].apply(extract_values)
df.dropna(subset=['compound'],inplace=True)

# Define date range filter
start_date = datetime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime('2023-01-01 23:59:59', '%Y-%m-%d %H:%M:%S').date()

start, end = st.sidebar.date_input('Date range:', value=(start_date, end_date))

filtered_data = df[(df['timestamp'] >= pd.to_datetime(start)) & (df['timestamp'] <= pd.to_datetime(end))]

# Group the data by month and calculate the mean compound score
grouped_data = filtered_data.groupby(pd.Grouper(key='timestamp', freq='M')).mean(numeric_only=True)['compound']
# Create a line chart changes over time
start_str = start.strftime('%Y-%m-%d %H:%M:%S')
end_str = end.strftime('%Y-%m-%d %H:%M:%S')
x_start = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').date()
x_end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').date()
fig = px.line(grouped_data, x=grouped_data.index, y='compound', title='Average Sentiment Score over time')
fig.update_layout(margin=dict(l=80, r=20, t=40, b=20),
    width=800,
    height=400,
    xaxis_title="Month",
    xaxis_range=[x_start, x_end])
st.title("Sentiment Trend by Month")
st.plotly_chart(fig)


@st.cache_data
def generate_token_encoder ():
    
    df = pd.read_csv("pages/Cleaned_combined_df.csv")

    df = df.dropna(subset=['sent_score', 'Cleaned_Tweet'])
    df['Cleaned_Tweet'] = df['Cleaned_Tweet'].astype(str)

    X = df['Cleaned_Tweet'].values
    y = df['sent_score'].values

    # Encoding the labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y, num_classes=3)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenization
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(X_train)

    # Convert the text to sequences
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Padding the sequences
    max_seq_length = 250
    X_train = pad_sequences(X_train, maxlen=max_seq_length)
    X_test = pad_sequences(X_test, maxlen=max_seq_length)

    # Model definition
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_seq_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(GRU(128))
    model.add(Dense(3, activation='softmax'))
    
    return tokenizer, encoder

# Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tokenizer, encoder = generate_token_encoder()


def lstm_sentiment_score(input_string, model, tokenizer, max_seq_length=250):
    # Preprocess the input string
    input_string = [input_string]
    tokenized_input = tokenizer.texts_to_sequences(input_string)
    padded_input = tf.keras.utils.pad_sequences(tokenized_input, maxlen=max_seq_length)

    # Predict the sentiment probabilities
    prediction = model.predict(padded_input)

    # Assign sentiment values
    sentiment_values = np.array([-1, 0, 1])

    # Calculate the weighted sum of sentiment values
    sentiment_score = np.dot(prediction, sentiment_values)

    return sentiment_score[0]


def predict_sentiment(input_string, model, tokenizer, encoder, max_seq_length=250):
 

    # Preprocess the input string
    input_string = [input_string]  # Convert the string to a list containing the string
    tokenized_input = tokenizer.texts_to_sequences(input_string)
    padded_input = tf.keras.utils.pad_sequences(tokenized_input, maxlen=max_seq_length)

    # Predict the sentiment
    prediction = model.predict(padded_input)

    # Decode the prediction
    sentiment_class = encoder.inverse_transform([np.argmax(prediction)])

    return sentiment_class[0]


model_path = "frontend_streamlit/pages/Bidirectional_LSTM.h5"
model = load_model(model_path)



# define css style for button
button_style = """
    <style>
    .stButton > button {
        width: 100%;
    }
    </style>
    """

st.markdown(button_style, unsafe_allow_html=True)


def check_input(tweet_input, model, tokenizer ):
    if tweet_input:
        with st.spinner("Generating score..."):
            score =lstm_sentiment_score(tweet_input, model, tokenizer )
            num = predict_sentiment(tweet_input, model,tokenizer, encoder )
        
        category = 'Negative'
        if num == 0.0:
            category = 'Neutral'
        elif num == 1.0:
            category = 'Positive'
            
        st.markdown('Your input: '+ tweet_input)
        st.markdown('is: '+ category)
        st.markdown('with score below')
        st.markdown(score)

    else:
        st.write('Please enter a tweet')

# define function to clear input
def clear_input():
    st.session_state['tweet_input'] = ''

# set page title and layout
st.title('LSTM sentiment model test')
st.subheader('This app generates sentiment score using given content or tweet, the score may be different due to different model, but the category should be the same')
tweet_input = st.text_area(label='Enter a tweet here to get sentiment score:', value='', key='tweet_input')

# create two columns for buttons
col1, col2 = st.columns(2)

# generate button
with col1:
    button_gene = st.button('Generate')
    
# clear input button
with col2:
    button_clear = st.button('Clear',on_click=clear_input)


if button_gene:
        check_input(tweet_input, model, tokenizer)




def visualise_sentiments(data):
    scores = []
    for sentence in data:
        # Compute sentiment scores for each word in the sentence
        word_scores = [sid.polarity_scores(word)["compound"] for word in sentence.split()]
        # Compute the average sentiment score for the sentence
        sentence_score = sid.polarity_scores(sentence)["compound"]
        scores.append([sentence_score] + word_scores)
    
    # Create DataFrame to store sentiment scores for each sentence
    df = pd.DataFrame(scores, columns=["Sentence Score"]+[word for word in data[0].split()])
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