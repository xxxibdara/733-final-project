#!/usr/bin/env python
# coding: utf-8


# Importing needed libraries
import sys, re
import pandas as pd
from datetime import datetime

tag_name = sys.argv[1]
folder_path = f"frontend_streamlit/pages/{tag_name}"

df = pd.read_csv(f'{folder_path}.csv')


def data_cleaning(df):
    df['user'] = df['user'].replace('[^a-zA-Z0-9]', '', regex=True)
    df['text'] = df['text'].replace('#', '', regex=True)
    df.dropna(subset=['text'], inplace=True)#3515
    df.drop_duplicates(subset=['text'], inplace=True)#3239
    return df

data_cleaning(df)

def clean_tweet(tweet):
    if not isinstance(tweet, str):
      tweet = str(tweet)
    tweet = re.sub(r'@\w+', '', tweet)  # Remove @username mentions
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)  # Remove URLs
    tweet = re.sub(r'#(\w+)', r'\1', tweet)  # Remove '#' from hashtags
    tweet = re.sub(r'\bRT :', '', tweet)  # Remove 'RT :' from retweets

    return tweet.strip()

df['text'] = df['text'].apply(clean_tweet)

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


def convert_timestamp(df):
    df = df.copy()
    df = df[df["timestamp"] != "0"]  # Remove rows where timestamp value is "0"
    df.loc[:, "timestamp"] = df["timestamp"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"))
    return df

df = convert_timestamp(df) #3238


df.to_csv(f'{tag_name}_clean.csv')
