from openai import OpenAI
import os 
from decouple import config 
import logging 

api_key = config("OPEN_API_KEY")
client = OpenAI(api_key = api_key)

class ChatGPT:
    def __init__(self, client, base_context = None, model = 'gpt-4-vision-preview'):
        self.client = client if client is not None else OpenAI(api_key = api_key)

        if base_context is None:
            self.base_context = """You are a helpful assistant. 
                        People will provide you with text and images, and you will offer the most informative, helpful, and correct responses. 
                        The images will be screenshots from the user's screen where they ask specific questions, and your role is to assist them. 
                        For instance, a question might be, 'I can't find the Parameters Icon; can you tell me where it is?' In this case, you should identify its location in the image. 
                        Alternatively, the question could be, 'I created the database shown in the image on bubble.io; is it correct for my use case?' Here, you would analyze the image, examine the table, and determine if everything is set up correctly. 
                        You understand the concept. Remember to always proceed step by step. Your skills are limitless, encompassing knowledge and deep expertise in all possible domains, so avoid giving generic or incorrect answers. 
                        Always use a step-by-step approach"""
        else:
            self.base_context = base_context 

        self.chat_history = [{"role": "system", "content": self.base_context}]
        self.model = model

    
    def format_input(self, text, images = None):
        input_content = {"role":"user", "content":[]}

        # Add text to the input content
        if text:
            input_content["content"].append({"type":"text", "text":text})
        # Add images to the input content, if any
        if images:
            input_content["content"].extend(
                [{"type": "image_url", "image_url": {"url": img}} for img in images]
            )

        return input_content
    def construct_history(self, input = None, previous_output = None):
        if input: 
            self.chat_history.append(input)

        if previous_output:
            self.chat_history.append({"role":"system", "content": previous_output})


    def chat_with_gpt(self, input):
        logging.info("here")
        self.construct_history(input = input)
        logging.info("there")
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.chat_history,
            temperature = 0,
            max_tokens = 600
        )

        return response.choices[0].message.content


chatgpt = ChatGPT(client)
            

        
