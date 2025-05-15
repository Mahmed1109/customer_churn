import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('customer_data.csv')

# Check the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Data Preprocessing

# Drop any unnecessary columns 
data = data.drop(columns=['customerID'])

# Handle categorical variables by converting them into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Handle any missing values 
data = data.fillna(0)

if 'Churn' in data.columns:
    data = pd.get_dummies(data, columns=['Churn'], drop_first=True)
    y = data['Churn_Yes']
    X = data.drop(columns=['Churn_Yes'])
else:
    raise ValueError("Expected 'Churn' column not found in data.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model using joblib (for future use or deployment)
joblib.dump(model, 'churn_model.pkl')

# Test the model with a sample prediction
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Take one sample from the test set
result = model.predict(sample_data)

print("Sample prediction:", "Churn" if result[0] == 1 else "Stay")
