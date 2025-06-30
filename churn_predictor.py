import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('customer_data.csv')
data = data.drop(columns=['customerID'])
data = data.replace(" ", 0)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
data = data.fillna(0)

if 'Churn' in data.columns:
    data = pd.get_dummies(data, columns=['Churn'], drop_first=True)
    y = data['Churn_Yes']
    X = data.drop(columns=['Churn_Yes'])
else:
    raise ValueError("Expected 'Churn' column not found in data.")

# Then encode all other categorical features
X = pd.get_dummies(X, drop_first=True)

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
sample_data = X_test.sample(5, random_state=42)
results = model.predict(sample_data)

# Print predictions
for i, pred in enumerate(results):
    print(f"Sample {i+1}: {'Churn' if pred == 1 else 'Stay'}")

#user input
# Ask the user if they want to predict for a new customer
user_choice = input("\nWould you like to predict churn for a new customer? (yes/no): ").strip().lower()
user_choice = input("\nWould you like to predict churn for a new customer? (yes/no): ").strip().lower()
if user_choice == 'yes':
    print("\n--- Please enter customer details ---")

    input_data = {
        'gender': input("Gender (Male/Female): "),
        'SeniorCitizen': int(input("Senior Citizen? (0 = No, 1 = Yes): ")),
        'Partner': input("Partner (Yes/No): "),
        'Dependents': input("Dependents (Yes/No): "),
        'tenure': int(input("Tenure (months): ")),
        'PhoneService': input("Phone Service (Yes/No): "),
        'MultipleLines': input("Multiple Lines (Yes/No/No phone service): "),
        'InternetService': input("Internet Service (DSL/Fiber optic/No): "),
        'OnlineSecurity': input("Online Security (Yes/No/No internet service): "),
        'OnlineBackup': input("Online Backup (Yes/No/No internet service): "),
        'DeviceProtection': input("Device Protection (Yes/No/No internet service): "),
        'TechSupport': input("Tech Support (Yes/No/No internet service): "),
        'StreamingTV': input("Streaming TV (Yes/No/No internet service): "),
        'StreamingMovies': input("Streaming Movies (Yes/No/No internet service): "),
        'Contract': input("Contract (Month-to-month/One year/Two year): "),
        'PaperlessBilling': input("Paperless Billing (Yes/No): "),
        'PaymentMethod': input("Payment Method (Electronic check/Mailed check/Bank transfer/Credit card): "),
        'MonthlyCharges': float(input("Monthly Charges: ")),
        'TotalCharges': float(input("Total Charges: "))
    }

    user_df = pd.DataFrame([input_data])
    user_df = pd.get_dummies(user_df, drop_first=True)
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    model = joblib.load('churn_model.pkl')
    prediction = model.predict(user_df)[0]

    print("\n📢 Prediction Result:")
    print("👉 The customer will **CHURN**." if prediction == 1 else "✅ The customer will **STAY**.")

else:
    print("\nNo user input prediction requested.")