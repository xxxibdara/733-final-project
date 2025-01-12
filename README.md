# Building a Popular Tweet: Tag Prediction and Sentiment Analysis

Demo Video: https://www.youtube.com/watch?v=lN1VXEbe8nU

# Menu 
- [Project title](#733-final-project)
- [Menu](#menu)
    - [Introduction](#introduction)
    - [Run the application](#run-the-application)
    - [Repo Structure](#repo-structure)
    - [Tech Stack](#tech-stack)
    - [Contributors](#contributors)


## Introduction
Social media platforms like Twitter have become one of the most popular sources of information and communication, especially among young people. Millions of tweets are posted every day on various topics ranging from news, politics, sports, entertainment, and more. These tweets contain a wealth of information, including people's opinions, thoughts, and sentiments about various topics.

Therefore, our project aims to address these challenges by developing a system that can perform sentiment analysis and tag prediction on tweets accurately. We believe that this project can provide valuable insights to companies, individuals, and researchers by allowing them to make informed decisions and discover new information.

## Run the application
### Before clone our repository
Our repository involes Git Large File (GLF), remember to install *git-lfs* in advance. 

### Run the application locally
With streamlit installed, enter our *frontend-streamlit* directory, run:
```
streamlit run homepage.py
```

### Check out our application on Streamlit Cloud
You can also check out our application on Streamlit Cloud with simply clicking the link below.
[streamlit cloud link](https://ziyaocui-733-final-project-frontend-streamlithomepage-i4lslq.streamlit.app/)

## Repo Structure 
- data_preprocessing: Preprocess Twitter data for further use.
- frontend_streamlit: Our frontend generated by Streamlit with interactive dashboards.
- scraping: Consists of source code for data collection and scraped data.
- sentiment_analysis: Contains data cleaning for sentiment analysis, along with two approach for sentiment score.
- tag_prediction: Includes model related files for tag prediction, such as model training, input data cleaning, saved model, interface for model prediction, etc.

## Tech Stack
- Data Collection: Selenium
- Data Processing: Spark, Pandas, Numpy, NLTK
- Model Development: PyTorch, Scikit-learn, Transformers, Keras
- Data Visualization: Streamlit, Matplotlib, Plotly, Wordcloud
- Deployment: Streamlit Cloud

## Contributors
- Xiner Qian @qian-x2
- Xiaoxiao Duan @stelladuan
- Zeye Gu @Simonmon06
- Ziyao Cui @ziyaocui
- Jingyi Huang @huangj20, @xxxibdara
