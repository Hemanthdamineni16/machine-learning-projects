import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset.csv")

# Display basic information about the dataset
print("Dataset information:")
print(data.info())


# Display first few rows of the dataset
print("First few rows of the dataset:")
data.head()

# Check for null values
print("Null values check:")
print(data.isnull().sum())

# Check languages present in dataset
print(" present Languages in dataset:")
print(data["language"].value_counts())


# Visualize distribution of languages
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="language")
plt.title("Distribution of Languages")
plt.xlabel("Language")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

unique_languages = data["language"].nunique()
print("Number of unique languages:")
print(unique_languages)

# Split data into features and labels
x = np.array(data["Text"])
y = np.array(data["language"])

# Convert text data into numerical format
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Multinomial Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Generate classification report
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Language detection from user input
user_input = input("Enter a Text: ")
user_data = cv.transform([user_input]).toarray()
output = model.predict(user_data)
print("Predicted Language:", output)