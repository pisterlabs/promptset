

import openai

import os

import pandas as pd

import time

import streamlit as st

import xgboost as xgb
loaded_model = xgb.Booster()
loaded_model.load_model('xgb_model.json')

openai.api_key = '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' #(Should be private)

def chatGPT(text):
  url = "https://api.openai.com/v1/completions"
  headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer YOUR-API-KEY",
  }
  data = {
  "model": "text-davinci-003",
  "prompt": text,
  "max_tokens": 4000,
  "temperature": 0.6,
  }
  response = requests.post(url, headers=headers, json=data)
  output = response.json()["choices"][0]["text"]

  return print(output)


def get_response(prompt, model="gpt-3.5-turbo"):

  messages = [{"role": "user", "content": prompt}]

  messages=[
      {"role": "system", "content": "You are a nice credit card fraud detector and assistant"},
      {"role": "user", "content": prompt},
  ]

  response = openai.ChatCompletion.create(

  model=model,

  messages=messages,

  temperature=0,

  )

  return response.choices[0].message["content"]


def get_predict_message(inputs):
    
    prediction = loaded_model.predict(inputs)
    prediction = int(prediction[0])

    predict_text= ""

    prompt = ""
    
    if prediction == 0:
        prompt += "Hi, kindly tell me congratulations that the credit card is not a fraud and tell me the ways toprevent credit card fraud from happening"
        predict_text+="This is not fraud"
    else:
        prompt += "Hi, kindly tell me sorry that the credit card is a fraud and tell me what to do after discovering that my credit card is fraud"
        predict_text+="This is a fraud"
    prompt_response = get_response(prompt)

    return predict_text, prompt_response

    



st.title("XGBoost with GPT Feedback")

# Collecting inputs
time = st.number_input("Time", value=67.000000)
v1 = st.number_input("V1", value=-1.494668)
v2 = st.number_input("V2", value=0.837241)
v3 = st.number_input("V3", value=2.628211)
v4 = st.number_input("V4", value=3.145414)
v5 = st.number_input("V5", value=-0.609098)
v6 = st.number_input("V6", value=0.258495)
v7 = st.number_input("V7", value=-0.012189)
v8 = st.number_input("V8", value=0.102136)
v9 = st.number_input("V9", value=-0.286164)
v10 = st.number_input("V10", value=1.198556)
v11 = st.number_input("V11", value=-0.550296)
v12 = st.number_input("V12", value=-0.106846)
v13 = st.number_input("V13", value=0.208014)
v14 = st.number_input("V14", value=-0.680510)
v15 = st.number_input("V15", value=0.507764)
v16 = st.number_input("V16", value=-0.260264)
v17 = st.number_input("V17", value=0.246631)
v18 = st.number_input("V18", value=0.008856)
v19 = st.number_input("V19", value=0.899416)
v20 = st.number_input("V20", value=-0.028352)
v21 = st.number_input("V21", value=-0.140047)
v22 = st.number_input("V22", value=0.355044)
v23 = st.number_input("V23", value=0.332720)
v24 = st.number_input("V24", value=0.718193)
v25 = st.number_input("V25", value=-0.219366)
v26 = st.number_input("V26", value=0.118927)
v27 = st.number_input("V27", value=-0.317486)
v28 = st.number_input("V28", value=-0.340783)
amount = st.number_input("Amount", value=28.280000)

columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


input_data = [[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]]
   
df = pd.DataFrame(input_data, columns=columns)  

data_dmatrix = xgb.DMatrix(data=df)

if st.button("Predict"):

    predict_text, prompt_response = get_predict_message(data_dmatrix)

    st.write(f"Prediction: {predict_text}")
    
    st.write(f"GPT Feedback: {prompt_response}")

if __name__ == '__main__':
    pass



#iodeajo@gmail.com