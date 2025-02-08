import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.linear_model import LogisticRegression  # Logistic Regression
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
data['Acceptance_category'] = pd.cut(data['AcceptanceRate'], bins=3, labels=['Low', 'Medium', 'High'])

# Define features (X) and target (y) with categorized acceptance rate
X = data.drop(['AcceptanceRate', 'Acceptance_category'], axis=1)  # Features (adjust if necessary)
y = data['Acceptance_category']  # Target: categorized acceptance rate

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models for SVM and Logistic Regression
svm_model = SVC(kernel='linear', random_state=42)  # Linear kernel SVM
logreg_model = LogisticRegression(max_iter=1000, random_state=42)  # Logistic Regression model

# Train and evaluate both models
models = {'SVM': svm_model, 'Logistic Regression': logreg_model}
accuracy_scores = {'SVM': [], 'Logistic Regression': []}

# Perform iterations to observe model performance over multiple runs
num_iterations = 10

for iteration in range(num_iterations):
    # Train and predict with SVM
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_scores['SVM'].append(accuracy_svm)

    # Train and predict with Logistic Regression
    logreg_model.fit(X_train, y_train)
    y_pred_logreg = logreg_model.predict(X_test)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    accuracy_scores['Logistic Regression'].append(accuracy_logreg)

# Print the accuracy scores for both models over multiple iterations
print("Accuracy scores for 10 iterations:")
for model_name, scores in accuracy_scores.items():
    print(f"{model_name}:")
    for i, score in enumerate(scores, 1):
        print(f"  Iteration {i}: {score:.4f}")

# Confusion Matrix for both models
y_pred_svm_final = svm_model.predict(X_test)
y_pred_logreg_final = logreg_model.predict(X_test)

print("\nConfusion Matrix for SVM:")
print(confusion_matrix(y_test, y_pred_svm_final))

print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logreg_final))
