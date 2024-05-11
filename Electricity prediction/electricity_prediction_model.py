import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read data from CSV file
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv")

# Display the first few rows of the dataset and its information
print(data.head())
data.info()

# Convert certain columns to numeric, handling errors with 'coerce'
data["ForecastWindProduction"] = pd.to_numeric(data["ForecastWindProduction"], errors='coerce')
data["SystemLoadEA"] = pd.to_numeric(data["SystemLoadEA"], errors='coerce')
data["SMPEA"] = pd.to_numeric(data["SMPEA"], errors='coerce')
data["ORKTemperature"] = pd.to_numeric(data["ORKTemperature"], errors='coerce')
data["ORKWindspeed"] = pd.to_numeric(data["ORKWindspeed"], errors='coerce')
data["CO2Intensity"] = pd.to_numeric(data["CO2Intensity"], errors='coerce')
data["ActualWindProduction"] = pd.to_numeric(data["ActualWindProduction"], errors='coerce')
data["SystemLoadEP2"] = pd.to_numeric(data["SystemLoadEP2"], errors='coerce')

# Check for and drop any rows with null values
print(data.isnull().sum())


# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Calculate correlations among numeric features
correlations = numeric_data.corr(method='pearson')

# Visualize correlations using a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

data = data.dropna()
# Define features (X) and target variable (y)
X = data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", 
          "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", 
          "ActualWindProduction", "SystemLoadEP2"]]
y = data["SMPEP2"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Ensure X_train is a DataFrame with proper column names
X_train = pd.DataFrame(X_train, columns=X.columns)

# Initialize and train Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)


# Define feature names
feature_names = ["Day", "Month", "ForecastWindProduction", "SystemLoadEA", 
                 "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", 
                 "ActualWindProduction", "SystemLoadEP2"]

# Define the features array
features = np.array([[10, 12, 54.10, 4241.05, 49.56, 9.0, 14.8, 491.32, 54.0, 4426.84]])

# Create a DataFrame with the input features and column names
features_df = pd.DataFrame(features, columns=feature_names)

# Feature Importance
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Predict electricity prices using the trained model
predicted_price = model.predict(features_df)
print("Predicted electricity price:", predicted_price)

