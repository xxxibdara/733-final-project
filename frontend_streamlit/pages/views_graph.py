# Description: This file contains the code for the views graph page

import pandas as pd
import streamlit as st
import plotly.express as px

# set the page title
st.title('Twitter data changes over time')

# set the page subtitle
st.subheader('This page displays the number of views, retweets, and comments for a given tag over time')

# create a sidebar for entering the tag name
st.sidebar.write('Our TOP 5 tags are: covid, news, technology, food, sports.')
tag = st.sidebar.text_input('Enter the tag name from our TOP5 list','covid')

# load data
df = pd.read_csv(f'./pages/{tag}.csv')

# convert timestamp column to datetime format, handling errors gracefully
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

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
metric = st.sidebar.selectbox('Metric:', ['num_views', 'num_retweets', 'num_comments'])

# create a date range slider for filtering the data by date
default_dates = [df['timestamp'].min(), df['timestamp'].max()]

start_date, end_date = st.sidebar.date_input('Date range:', default_dates, min_value=df['timestamp'].min(), max_value=df['timestamp'].max())

start_date = pd.Timestamp(start_date, tz='UTC')
end_date = pd.Timestamp(end_date, tz='UTC')
filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]


# display the chart based on the selected metric and date range
display_chart(metric)
