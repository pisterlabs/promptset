import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

print(openai.api_key)

import time
from pprint import pprint

import gradio as gr

# from log import logger


# Chatbot class
class Chatbot():
    def __init__(self):
        self.full_chat_history = []
        self.current_chat_history = []
        self.default_message = {"role": "system", "content": "You are a helpful and firendly assistant"}
        self.messages = [self.default_message]
        self.len_chat_history = 4
        self.role = "You are a helpful and firendly assistant"
        self.query_delimiter = "####"
        self.context_delimiter = "####"
        self.model = "gpt-3.5-turbo"
        self.temperature = 0
        self.max_tokens = 350

        self.system_template_prompt = f"""
            You are given a query which is enclosed by {self.query_delimiter}.
            Your role is to assist customers by generating detailed prompts for the type of clothes they are looking for based on the query.
            Only generate prompts for one type of clothing at a time. 
            For example, if the query is "I am looking for a pair of jeans and a shirt", then generate prompts for jeans and shirt separately.
            You can use the following template to generate prompts for the type of clothing the customer is looking for.
            """
        
        self.fallback_message = "Some error occurred. Please try again later."

    # Set the document parameters
    def set_doc_params(self,doc_params):
        self.doc_params = doc_params

    def set_model(self,model):
        self.model = model

    def set_temperature(self,temperature):
        self.temperature = temperature
    
    def get_conversation(self):
        conversation = ""
        for i in range(0,len(self.current_chat_history),2):
            conversation += "Human: "+self.current_chat_history[i]['content'] + '\n'
            conversation += "AI: "+self.current_chat_history[i+1]['content'] + '\n'
        return conversation

    def get_response(self,query,faq = True, model = None,temperature = None,return_sources = False, debug = False):
        faq_triggered = False
        if model is None: model = self.model
        if temperature is None: temperature = self.temperature
        self.query = query
        if debug:
            pprint(query)
        query_prompt = f'''
            {self.system_prompt_template}

            Query:
            {self.query_delimiter}{query}{self.query_delimiter}
            '''
        self.messages.append({"role": "user", "content": f"{query_prompt}"})
        completion = openai.ChatCompletion.create(
            api_key = os.getenv("OPENAI_API_KEY"),
                        model=self.model,
                        messages=self.messages,
                        temperature=self.temperature,
                        max_tokens = self.max_tokens                       
 )

        response = completion.choices[0]['message']['content']
        self.messages.pop()
        if debug:
            pprint(completion)
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        elif "A:" in response:
            response = response.split("A:")[-1].strip()
        elif "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        self.response = response
        self.full_chat_history.append({"role": "user", "content": f"{query}"})
        self.full_chat_history.append({"role": "assistant", "content": response})
        if len(self.current_chat_history) == (2*self.len_chat_history):
            self.current_chat_history = self.current_chat_history[2:]
        self.current_chat_history.append({"role": "user", "content": f"{query}"})
        self.current_chat_history.append({"role": "assistant", "content": response})
        self.messages = self.current_chat_history
        return response

    # Function to clear the chat history
    # Clears the current chat history
    def clear_chat(self):
        self.current_chat_history.clear()
        self.messages = [self.default_message]
        self.full_chat_history = []

    # Function to launch the chatbot
    # Uses gradio to launch the chatbot
    def launch(self,share = False,show_sources= False, debug = False):
        with gr.Blocks() as demo:

            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")            
            def user(user_message, history):
                return "", history + [[user_message, None]]
            def bot(history):
                if show_sources:
                    response,sources = self.get_response(history[-1][0],debug = debug,return_sources=True)
                else:
                    response = self.get_response(history[-1][0],debug = debug,return_sources=False)
                if show_sources:
                    urls = ""
                    for url in sources:
                        urls += url
                        urls += "\n"
                    bot_message = response + "\n" + urls
                else: 
                    bot_message = response
                history[-1][1] = ""
                for character in bot_message:
                    history[-1][1] += character
                    time.sleep(0.01)
                    yield history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(self.clear_chat,  None, chatbot, queue=False)
        demo.queue()
        demo.launch(share=share)
