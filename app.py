import uvicorn
from fastapi import FastAPI
from Crop_Recommender import Crop_Recommender
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello'}

@app.get('/{name}')
def get_name(name: str):
    return {'Enter your crop details': f'{name}'}

@app.post('/predict')
def predict(data: Crop_Recommender):
    data = data.dict()
    crop_names = ['rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange',
              'apple', 'muskmelon', 'watermelon', 'grapes', 'mango', 'banana',
              'pomegranate', 'lentil', 'blackgram', 'mungbean', 'mothbeans',
              'pigeonpeas', 'kidneybeans', 'chickpea', 'coffee']

    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']
    
    prediction = classifier.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = crop_names[prediction[0]]
    
    return {"predicted_crop": predicted_crop}

# Run the FastAPI app with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    