#Data Science ProjectWork

This repository contains various data science projects, showcasing different techniques such as regression, sentiment analysis, and time series forecasting.

#Projects

#1. Linear Regression on Salary Dataset
Objective:
Predict employee salaries based on years of experience using a linear regression model.

Steps:

Libraries Used: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn.
Dataset: Salary dataset with columns for years of experience and salary.
Preprocessing: Cleaned data by removing irrelevant columns and performing exploratory data analysis (EDA).
Model: Built and evaluated a linear regression model.
Performance Metrics: MAE (~6286.45), MSE (~49.83M), R-squared (~0.902).
Key Takeaways:

Strong linear relationship between years of experience and salary.
High R-squared value indicating good model performance.

#2. Sentiment Analysis on Social Media Data
Objective:
Analyze sentiment from social media posts using natural language processing (NLP) techniques and visualize public opinion.

Features:

Sentiment classification using VADER.
Text cleaning and hashtag analysis.
Visualizations of sentiment distribution by platform, country, and over time.
Engagement analysis (likes, retweets).
Steps:

Preprocessing: Cleaned text and extracted hashtags.
Sentiment Analysis: Applied VADER sentiment analysis.
Data Visualization: Visualized sentiment trends, engagement metrics, and platform-wise sentiment.
Requirements:
Python 3.8+, Libraries: nltk, pandas, matplotlib, seaborn, textblob, wordcloud.

Installation:

git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
pip install nltk pandas matplotlib seaborn textblob wordcloud

#3. Time Series Forecasting for Stock Prices
Objective:
Forecast Meta's stock prices using ARIMA and LSTM models.

Dataset:
Meta's stock data from 2020-2024 with columns like open, high, low, close, adjusted close, and volume.

Steps:

Preprocessing: Cleaned and split data into training (80%) and testing (20%).
Models:
ARIMA: Trained with order (5, 1, 0), evaluated with RMSE (~118.46) and MAE (~87.36).
LSTM: Scaled data with MinMaxScaler, used a sliding window approach for training.
Performance:

ARIMA: RMSE = 118.46, MAE = 87.36.
LSTM: Model still under training, expected superior performance.
Installation:

git clone https://github.com/yourusername/stock-forecasting.git
cd stock-forecasting
pip install -r requirements.txt

#4. Exploratory Data Analysis (EDA) on Titanic Dataset
Objective:
Perform exploratory data analysis (EDA) on the Titanic dataset to understand key factors influencing passenger survival.

Dataset:
The Titanic dataset contains information about passengers on the Titanic, including their demographics (age, sex, class, etc.) and whether they survived.

Steps:

Libraries Used: Pandas, Matplotlib, Seaborn.
Preprocessing: Cleaned the dataset by handling missing values and performing feature engineering (e.g., filling missing age values, creating new features like "Family Size").
Visualization:
Distribution of passengers by age, sex, and class.
Correlation between survival and different features.
Survival rate by class, gender, and age group.
Key Insights:

Women had a higher survival rate than men.
Passengers in first class had a higher survival rate.
Younger passengers had a better chance of survival.
Code Example for EDA:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Data cleaning (e.g., filling missing values)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Visualization: Survival by Sex
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Visualization: Survival by Class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')
plt.show()

# Correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
Conclusion

These projects demonstrate the application of various machine learning and statistical techniques. The linear regression model predicts salaries, sentiment analysis uncovers insights from social media, time series forecasting predicts stock prices, and EDA on the Titanic dataset provides insights into passenger survival. Future improvements for these projects include experimenting with more advanced models, hyperparameter tuning, and real-time deployment.
