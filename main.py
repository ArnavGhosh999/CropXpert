import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Prepare the features and labels
X = df[['N', 'P', 'K', 'temperature', 'ph', 'rainfall', 'humidity']]
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Define input boxes for user input
st.title('Crop Recommendation System')

# Input boxes with default values and type conversion
N = st.text_input('N (Nitrogen)', '0')
P = st.text_input('P (Phosphorus)', '0')
K = st.text_input('K (Potassium)', '0')
temperature = st.text_input('Temperature (°C)', '0.0')
humidity = st.text_input('Humidity (%)', '0')
rainfall = st.text_input('Rainfall (mm)', '0')
ph = st.text_input('pH', '0.0')

# Convert inputs to appropriate types
try:
    N = float(N)
    P = float(P)
    K = float(K)
    temperature = float(temperature)
    humidity = float(humidity)
    rainfall = float(rainfall)
    ph = float(ph)
except ValueError:
    st.error('Please enter valid numerical values')
    st.stop()

# Predict the crop
prediction = classifier.predict([[N, P, K, temperature, ph, rainfall, humidity]])
st.write(f'**Predicted Crop:** {prediction[0]}')

# Optionally, show feature importances
if st.checkbox('Show Feature Importances'):
    feature_importances = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.bar_chart(feature_importance_df.set_index('Feature'))
