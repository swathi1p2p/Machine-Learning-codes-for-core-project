import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/conference_paper_data.csv"  # Update with your dataset path
data = pd.read_csv(file_path)

# Print the first few rows to understand the structure of the dataset
print("Original Dataset:")
print(data.head())

# Check unique values in the 'Conference' column (or any relevant column for categorization)
print("Unique values in Conference column:", data['Conference'].unique())

# Filter the dataset if you want to focus on specific conferences or types of papers (optional)
# Example: You can filter the dataset based on 'Conference' if you need only specific conferences
# paper_data = data[data['Conference'] == 'SomeConference']

# Convert categorical columns to numerical using LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column in the dataframe (if there are any)
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str).str.strip())

# Define features (X) and target (y) from the dataset
X = data.drop('AcceptanceRate', axis=1)  # Replace 'AcceptanceRate' with the actual target column name
y = data['AcceptanceRate']               # Target: Acceptance Rate

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred = dt_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Decision Tree Mean Squared Error (MSE): {mse:.4f}")
print(f"Decision Tree R-squared (R²): {r2:.4f}")

# Calculate accuracy of predictions based on actual acceptance rate
accuracy = 100 * (1 - (np.abs(y_test - y_pred) / y_test))
accuracy_df = pd.DataFrame({
    'Actual Acceptance Rate': y_test,
    'Predicted Acceptance Rate': y_pred,
    'Accuracy': accuracy
})

# Select top 10 predictions based on actual acceptance rate
top_10_predictions = accuracy_df.nlargest(10, 'Actual Acceptance Rate')

print("Top 10 Acceptance Rate Predictions:")
print(top_10_predictions)
