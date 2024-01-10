import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key=os.environ.get("API_KEY")

def get_gpt_response(system_prompt,user_prompt):
  response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106", temperature = 0,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}
                ])
  gpt_response =  response["choices"][0]["message"]["content"]
  return(gpt_response)