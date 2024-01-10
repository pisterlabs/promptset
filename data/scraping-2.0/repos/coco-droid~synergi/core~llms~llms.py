import os
import openai
from params.config import APIKeyManager
from llms.jurassic import Jurassic
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage  
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from  llms.clarifai_pipeline import ClarifaiAPI
import requests

openai.api_key = APIKeyManager().get_api_key('openai_key')
print(openai.api_key)
class Model:

  MODELS = {
    'claude': ChatAnthropic,
    'gpt4': ChatOpenAI,
    'gpt3': ChatOpenAI, 
    'jurassic': Jurassic,
    "gpt4-clarifai":ClarifaiAPI,
    "claude-clarifai":ClarifaiAPI
  }

  def __init__(self, model_name, **kwargs):
    self.model_name = model_name
    
    if model_name.lower() in self.MODELS:
      model_class = self.MODELS[model_name.lower()]

      if model_name.lower() == 'claude':
        self.model = model_class(model='claude-2')

      elif model_name.lower() == 'jurassic':
        self.model = model_class(kwargs['api_key'])
      
      elif model_name.lower() in ['gpt3', 'gpt4']:
        #openai.api_key = os.getenv("OPENAI_API_KEY")

        print("you have openai api key")
    else:
      raise ValueError(f"Invalid model name: {model_name}")

    self.master_prompt = kwargs.get('master_prompt', None)

  def generate_text(self, user_input):

    prompt = user_input
    if self.master_prompt:
      prompt = f"{self.master_prompt} {prompt}"
      print(f"Prompt: {prompt}")
    
    try:
      if self.model_name.lower() == 'jurassic':
        response = self.model.generate(prompt)  

      elif self.model_name.lower() == 'claude':
        messages = [HumanMessage(content=prompt)]
        response = self.model.predict_messages(messages)[0].content
      elif self.model_name.lower() == 'claude-clarifai':
            response=self.generate_text_with_clarifai('claudev2',prompt).data.text.raw
            print(response)
      elif self.model_name.lower() == 'gpt4-clarifai':
            response=self.generate_text_with_clarifai('gpt4',prompt).data.text.raw
            print(response)
      elif self.model_name.lower() in ['gpt3', 'gpt4']:
        print("gpt3")
        print(self.master_prompt)
        print(openai.api_key)
        openai.api_key=APIKeyManager().get_api_key('openai_key')
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-16k",
          messages=[
            {"role": "system", "content": self.master_prompt},
            {"role": "user", "content": user_input}
          ])
        response=response.choices[0].message.content
        print(response)
    except Exception as e:
      print(f"Error: {e}")
      return "Désolé, erreur lors de la génération de texte."

    return response
  def generate_text_with_image(self,image_url,image_bytes):
      clarifai = ClarifaiAPI('blip2')
      if image_url:
        clarifai_response = clarifai.predict_images(image_url)
        return clarifai_response
      elif image_bytes:
        clarifai_response = clarifai.predict_images(image_bytes)
        return clarifai_response
      else:
        return "Désolé, erreur lors de la génération de texte."
  def generate_text_with_clarifai(self,model,text):
    clarifai = ClarifaiAPI(model)
    clarifai_response = clarifai.predict_text_raw(text)
    return clarifai_response
  def predict_with_video(self,video_url):
    clarifai = ClarifaiAPI()
    clarifai_response = clarifai.predict_video(video_url)
    return clarifai_response
  def generate_image_with_text(self,text):
    clarifai = ClarifaiAPI('stable-diffusion-xl')
    clarifai_response = clarifai.generate_image_with_text(text)
    return clarifai_response
  def generate_dalle(self,text):
      response = openai.Image.create(
          prompt=text,
          n=1,
          size="1024x1024")
      return  response['data'][0]['url']
  def generate_image_dalle(self,text):
      clarifai = ClarifaiAPI('dalle')
      clarifai_response = clarifai.generate_image_with_text(text)
      return clarifai_response
  def  generate_text_with_GIT(self,image):
       API_URL = "https://api-inference.huggingface.co/models/microsoft/git-base"
       headers = {"Authorization": "Bearer"+APIKeyManager().get_api_key('hugging_face')}
       #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"}
       with open(filename, "rb") as f:
          data = f.read()
       response = requests.post(API_URL, headers=headers, data=data)
       return response.json()
  def encode(self,text):
    clarifai = ClarifaiAPI('embedada')
    clarifai_response = clarifai.generate_text_embedding(text)
    return clarifai_response

