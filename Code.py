import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# Load the data
df_train = pd.read_csv('D:\INTERNSHIP\PROJECT\HEALTH CARE ANALYTICS\Train.csv')
df_test = pd.read_csv('D:\INTERNSHIP\PROJECT\HEALTH CARE ANALYTICS\Test.csv')

# Check the first few rows and data info
print(df_train.head())
print(df_train.info())

# Check for missing values
print(df_train.isnull().sum())

# Remove duplicates
df_train.drop_duplicates(inplace=True)

# Check the data types and unique values in each column
for column in df_train.columns:
    print(f"Data Type of {column}: {df_train[column].dtype}")
    print(f"Total Unique values in {column}: {df_train[column].nunique()}")
    if df_train[column].dtype in ['int64', 'float64']:
        print(f"Minimum value: {df_train[column].min()},   Maximum value: {df_train[column].max()}")
    print(f"Unique values in {column}: {df_train[column].unique()[:10]}\n")

# Separate numerical and categorical columns
num_col_train = df_train.select_dtypes(include=np.number).columns
cat_col_train = df_train.select_dtypes(include=['object', 'category']).columns

# Visualize Numerical columns
for col in num_col_train:
    print(col)
    print(f"Minimum value: {df_train[col].min()},   Maximum value: {df_train[col].max()}")  
    print('Skew :', round(df_train[col].skew(), 2))
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    df_train[col].hist(grid=False)
    plt.ylabel('Count')
    plt.title(col)
    plt.subplot(1, 2, 2)
    plt.title(col)
    sns.boxplot(x=df_train[col])
    plt.show()

# Visualize Categorical columns
for col in cat_col_train:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df_train)
    plt.title(f'Distribution of {col}')
    plt.show()

# Data preprocessing and feature engineering can be performed here

# Split the data into features and target variable
X = df_train.drop(columns=['Stay'])  
y = df_train['Stay']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define and train the Logistic Regression model
imputer = SimpleImputer(strategy="mean")  # Replace NaNs with mean
model = LogisticRegression()

pipeline = Pipeline([
    ("imputer", imputer),
    ("model", model)
])

pipeline.fit(X_train, y_train)
# Evaluate the model
y_pred = pipeline.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Make predictions on the test data
X_test_encoded = pd.get_dummies(df_test)  # Encode test data
predictions = pipeline.predict(X_test)

# Create a DataFrame for submission
if len(predictions) != len(df_test):
    # Handle missing predictions (choose one approach):
    missing_value = -1
    predictions = np.pad(predictions, (0, len(df_test) - len(predictions)), mode='constant', constant_values=missing_value)

submission = pd.DataFrame({'case_id': df_test['case_id'], 'Stay': predictions})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)