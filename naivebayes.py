import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/conference_paper_data.csv"  # Update with your dataset path
data = pd.read_csv(file_path)

# Print the first few rows to understand the structure of the dataset
print(data.head())

# Convert categorical columns to numerical using LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column in the dataframe
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column is categorical
        data[column] = label_encoder.fit_transform(data[column].astype(str).str.strip())

# Assuming 'AcceptanceRate' is continuous, categorize it into classes (Low, Medium, High)
# You can define custom bins based on your data's distribution
data['Acceptance_category'] = pd.cut(data['AcceptanceRate'], bins=3, labels=['Low', 'Medium', 'High'])

# Define features (X) and target (y) with categorized acceptance rate
X = data.drop(['AcceptanceRate', 'Acceptance_category'], axis=1)  # Features (adjust if necessary)
y = data['Acceptance_category']  # Target: categorized acceptance rate

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model (GaussianNB)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# Evaluate the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Classification Accuracy: {accuracy:.4f}")

# Confusion matrix to understand the classification results
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Perform iterations to observe model performance over multiple runs
num_iterations = 10
accuracy_scores = []

for iteration in range(num_iterations):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

print("Accuracy scores for 10 iterations (Naive Bayes Classifier):")
for i, score in enumerate(accuracy_scores, 1):
    print(f"Iteration {i}: {score:.4f}")
