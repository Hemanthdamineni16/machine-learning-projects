import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn import metrics


# Load the dataset
data = pd.read_csv('LoanApprovalPrediction.csv')


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

# Fill missing values
data.ffill(inplace=True)

# Convert categorical variables into numerical
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])


# Visualize the target variable distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
data['Loan_Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Loan Approval Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')

# Visualize Applicant Income distribution
plt.subplot(1, 2, 2)
sns.histplot(data['ApplicantIncome'], kde=True, color='green')
plt.title('Applicant Income Distribution')
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Visualize the impact of Credit History on Loan Approval
plt.figure(figsize=(8, 6))
sns.countplot(x='Credit_History', hue='Loan_Status', data=data, palette='pastel')
plt.title('Impact of Credit History on Loan Approval')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.show()

# Boxplot for Loan Amount by Loan Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='LoanAmount', y='Loan_Status', data=data, hue='Loan_Status', palette='pastel', orient='h', linewidth=2)
plt.title('Loan Amount by Loan Approval Status')
plt.xlabel('Loan Amount')
plt.ylabel('Loan Status')
plt.legend(title=None)
plt.show()


# Pairplot for numeric features
sns.pairplot(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])
plt.show()

# Data analysis
# Loan Approval rate by Gender
gender_approval_rate = data.groupby('Gender')['Loan_Status'].value_counts(normalize=True).unstack()
print("Loan Approval Rate by Gender:")
print(gender_approval_rate)

# Loan Approval rate by Education
education_approval_rate = data.groupby('Education')['Loan_Status'].value_counts(normalize=True).unstack()
print("\nLoan Approval Rate by Education:")
print(education_approval_rate)

# Splitting the dataset into features and target variable
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Making predictions
y_pred = classifier.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# Label Encoding for object datatype columns
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
    data[col] = label_encoder.fit_transform(data[col])


# Heatmap for correlation matrix
plt.figure(figsize=(12, 6)) 
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Catplot for Gender vs Married vs Loan_Status
sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.title('Gender, Married and Loan_Status')
plt.show()


# Get the list of categorical columns
object_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

plt.figure(figsize=(6, 10))
index = 1

# Iterate through the categorical columns and create the bar plots
for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(6, 2, index)
    plt.ylabel(f'{col.capitalize()} Counts', fontsize=7)

    plt.text(0.5, 1.02, f'0 to {y.max()}', transform=plt.gca().transAxes, fontsize=10, ha='center')
    plt.xticks(rotation=0, ha='right')
    sns.barplot(x=list(y.index), y=y.values, hue=y.index, palette='Set2', width=0.8, legend=False)
    
    # Set the y-axis limits and ticks
    plt.ylim(bottom=0, top=y.max() * 1.1)
    plt.yticks(ticks=range(0, int(y.max() * 1.1), 100))
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    index += 1

plt.tight_layout()
plt.show()


# Assuming X and Y are defined from your DataFrame
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

# Convert DataFrame to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Define your classifiers
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
svc = SVC()
lc = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence warning

# Creating pipelines to handle missing values and scaling for RandomForestClassifier, KNeighborsClassifier, and LogisticRegression
rfc_pipeline = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier())
knn_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), KNeighborsClassifier())
svc_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), SVC())
lc_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000))

classifiers = [(rfc_pipeline, "RandomForestClassifier"), 
               (knn_pipeline, "KNeighborsClassifier"), 
               (svc_pipeline, "SVC"), 
               (lc_pipeline, "LogisticRegression")]

# Training and evaluating the classifiers
for clf, name in classifiers:
    clf.fit(X_train, Y_train)
    Y_pred_train = clf.predict(X_train)
    train_accuracy = metrics.accuracy_score(Y_train, Y_pred_train)
    print(f"Training Accuracy score of {name} = {train_accuracy * 100:.2f}%")

print("\n") 

# Making predictions on the testing set
for clf, name in classifiers:
    Y_pred_test = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(Y_test, Y_pred_test)
    print(f"Testing Accuracy score of {name} = {test_accuracy * 100:.2f}%")
