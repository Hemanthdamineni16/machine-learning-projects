import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk


# Read the data
data = pd.read_csv("dataset user reviews.csv")

# Print DataFrame information
print("Data Information:")
print(data.info())

# Print DataFrame shape
print("\nDataFrame Shape:")
print(data.shape)

# Print first few rows of the DataFrame
print("\nFirst few rows:")
print(data.head())


# Print last few rows of the DataFrame
print("\nLast few rows:")
print(data.tail())


# Descriptive statistics for numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(data.describe())


# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Drop null values
print("\nnull Values check:")
data = data.dropna()
print(data.isnull().sum())


# Sentiment analysis using NLTK's Vader
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Translated_Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Translated_Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Translated_Review"]]

# Distribution of sentiments
plt.figure(figsize=(10, 6))
sns.histplot(data['Sentiment'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 6))
data['Sentiment'].value_counts().plot(kind='bar', color=['lightgreen', 'lightcoral', 'skyblue'])
plt.title('Sentiment Counts')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Word cloud of reviews
all_reviews = ' '.join(data['Translated_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Reviews")
plt.axis('off')
plt.show()

# Pie chart for sentiment distribution
sentiment_distribution = data['Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral', 'lightgreen'])
plt.title("Sentiment Distribution")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Scatter plot for sentiment analysis
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Sentiment_Polarity', y='Sentiment_Subjectivity', hue='Sentiment', data=data, edgecolor='white', palette="twilight_shifted_r")
plt.title("Google Play Store Reviews Sentiment Analysis")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Sentiment Subjectivity")
plt.legend(title="Sentiment")
plt.grid(True)
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['Sentiment_Polarity', 'Sentiment_Subjectivity', 'Positive', 'Negative', 'Neutral']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Additional analysis
positive_reviews = data[data['Sentiment'] == 'Positive']
negative_reviews = data[data['Sentiment'] == 'Negative']
neutral_reviews = data[data['Sentiment'] == 'Neutral']

print("\nAdditional Analysis:")
print("Number of Positive Reviews:", len(positive_reviews))
print("Number of Negative Reviews:", len(negative_reviews))
print("Number of Neutral Reviews:", len(neutral_reviews))
print("\nAverage Positive Sentiment Polarity:", positive_reviews['Sentiment_Polarity'].mean())
print("Average Negative Sentiment Polarity:", negative_reviews['Sentiment_Polarity'].mean())
print("Average Neutral Sentiment Polarity:", neutral_reviews['Sentiment_Polarity'].mean())
