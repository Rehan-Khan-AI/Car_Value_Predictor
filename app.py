import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load the car data (replace with your actual data path)
car_data = pd.read_csv("raw_car_data.csv")

# Feature engineering (consider adding more features based on your data)
features = car_data.drop("PRICE", axis=1)
target = car_data["PRICE"]

# Preprocessing
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, random_state=42
)

# Create the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app

st.title("Cars Price Prediction App")

# User input for new model year
new_model_year = st.number_input("Enter Model Year (e.g., 2025):", min_value=2000)

# Create a DataFrame for the new model
new_model = pd.DataFrame({"MODEL": [new_model_year]})

# Preprocess the new model data
new_model_transformed = scaler.transform(new_model)

# Make prediction
predicted_price = model.predict(new_model_transformed)[0]

# Display prediction with informative text
st.success(f"Predicted price for a car model year {new_model_year} is: ${predicted_price:.2f}")

# Optional: Display additional insights (consider adding model evaluation metrics)
# st.write("**Model Evaluation** (Add metrics like R-squared, MAE, etc.)")

st.stop()
