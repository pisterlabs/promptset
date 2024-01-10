import openai
from transformers import OpenAiAgent
import os

pswd =  os.environ["OPENAI_API_KEY"]
agent = OpenAiAgent(model="text-davinci-003", api_key=pswd)
 
dogs = agent.run("Generate an images of playing dogs")
dogs