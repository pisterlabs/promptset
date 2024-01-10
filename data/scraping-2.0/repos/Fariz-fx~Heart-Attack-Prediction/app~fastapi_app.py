import datetime
from fastapi import FastAPI, HTTPException
import logging
from openai_integration import generate_response
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
#import pickle
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

app = FastAPI()

# Load the trained model 
model = joblib.load('model/Heart_Attack_KNN_model.joblib')
scaler = joblib.load('model/Heart_Attack_KNN_scaler.joblib') 

class HeartAttackPredictionRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    
def preprocess(data: HeartAttackPredictionRequest):
    df = pd.DataFrame(data.dict(), index=[0])  # fixed code error
    #df = pd.DataFrame(data)
    return scaler.transform(df)

@app.post("/predict_chatgpt")
def predict_heart_attack(data: HeartAttackPredictionRequest):
    try:
        # Logging essential information
        logger.info(f"Data Submitted: {now}")
        prompt = f"Predict the probability of a heart attack with the following input:\n\nAge: {data.age}\nSex: {data.sex}\nChest Pain Type: {data.cp}\nTrestbps: {data.trestbps}\nChol: {data.chol}\nFBS: {data.fbs}\nRestecg: {data.restecg}\nThalach: {data.thalach}\nExang: {data.exang}\nOldpeak: {data.oldpeak}\nSlope: {data.slope}\nCA: {data.ca}\nThal: {data.thal}\n\nRisk factors for heart attack are: smoking, high blood pressure, high cholesterol, obesity, diabetes, sedentary lifestyle, and family history."
        prediction = generate_response(prompt)
        logger.info(f"ChatGPT Prompting: {prediction}")
        return {"prediction": prediction}
            #return {"prediction": prediction.item()}  # Return prediction as a single value

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": "Failed to process the request."}

@app.post("/predict_custom_model")
def predict_heart_attack_custom_model(data: HeartAttackPredictionRequest):
    try:
        # Logging essential information
        logger.info(f"Data Submitted: {now}")

        # preprocessing
        data_preprocessed = preprocess(data)

        # Make prediction
        prediction = model.predict(data_preprocessed)
        prediction_probability = model.predict_proba(data_preprocessed)
        prediction_probability_percentage = prediction_probability[0][1] * 100
        logger.info(f"Prompting: {prediction} \n Prediction Probability: {prediction_probability}")
        message = 'The probability of the person having a heart attack is {:.2f}%.'.format(prediction_probability_percentage)
        return {
            "message": message,
            "Prediction_Probability_Percentage": prediction_probability_percentage,
        }
            #{"prediction": int(prediction[0])}  # Return prediction as a single value

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": "Failed to process the request."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
