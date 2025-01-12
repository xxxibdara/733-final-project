# Description: This file contains the code for the views graph page

import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# set the page title
st.title('Twitter data changes over time')

# set the page subtitle
st.subheader('This page displays the number of views, retweets, and comments for a given tag over time')

# create a sidebar for entering the tag name
st.sidebar.title('Top 5 tags')
st.sidebar.write('covid, news, technology, food, sports.')
tag = st.sidebar.selectbox('Select a tag:', ['covid', 'news', 'technology', 'food', 'sports'])

# load data
# In prod env use this.
df = pd.read_csv(f'frontend_streamlit/pages/{tag}_clean.csv')
# In dev env use this.
# df = pd.read_csv(f'pages/{tag}_clean.csv')

# convert timestamp column to datetime format, handling errors gracefully
df['timestamp'] = pd.to_datetime(df['timestamp'])

# drop rows with NaN values
df = df.dropna(subset=['timestamp', 'num_views', 'num_retweets', 'num_comments'])

# create a function to display the chart
def display_chart(metric):
    fig = px.bar(filtered_df, x='timestamp', y=metric, title=f'{metric} of #{tag} over time')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
    width=800,
    height=400)
    st.plotly_chart(fig)

# create a sidebar for selecting the metric to display
st.sidebar.title('Select a metric to display')
metric = st.sidebar.selectbox('Metric:', ['num_retweets', 'num_comments', 'num_views'])

# create a date range slider for filtering the data by date
start_date = datetime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime('2023-01-01 23:59:59', '%Y-%m-%d %H:%M:%S').date()

start, end = st.sidebar.date_input('Date range:', value=(start_date, end_date))
start_str = pd.to_datetime(start)
end_str = pd.to_datetime(end)

filtered_df = df[(df['timestamp'] >= start_str) & (df['timestamp'] <= end_str)]


# display the chart based on the selected metric and date range
display_chart(metric)
