"""from service.get_gpt_service import GptService
cur_service = GptService()
response_content = cur_service.get_gpt_response(question)
print(type(response_content))
print(response_content["choices"])"""
import os
import openai
from dotenv import dotenv_values
env_vars = dotenv_values('utils/.env')
openai.api_key = env_vars["apikey"]
data = [int(i) for i in [1.0, 23.0, 1.0, 0.0, 1.0, 76, 87, 35]]
query = "Give recommendations to a 23-year-old male with hypertension, heart disease, a smoking history, a BMI of 76, HbA1c level of 87, and a blood glucose level of 35."
query = "Give recommendations to a {age}-year-old {gender} with {hypertension}, {heart_disease}, {smoking_history}, a BMI of {BMI}, HbA1c level of {HbA1c}, and a blood glucose level of {glucose}. Make you repsonse in second person pronoun.'".format(age = data[1], 
                                                                                                                                                                                                            gender = "male" if data[0] == 1.0 else "female", 
                                                                                                                                                                                                            hypertension = "hypertension" if data[2] == 1.0 else "no hypertension", 
                                                                                                                                                                                                            heart_disease = "heart diease" if data[3] == 1.0 else "no heart diease", 
                                                                                                                                                                                                            smoking_history = "smoking history" if data[4] == 1.0 else "no smoking history",
                                                                                                                                                                                                            BMI = int(data[5]), HbA1c = data[6], glucose = data[7])
question = [
    {"role": "system", "content": "You are a health consultant specialized in diabetes."},
    {"role": "user", "content": query}
]
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=question,
  temperature=0,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response['choices'][0]['message']['content'])
