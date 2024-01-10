#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import panel as pn
import re
import ipywidgets as widgets
pn.extension()


import os
import openai
from dotenv import load_dotenv,find_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain import output_parsers
from langchain.output_parsers import StructuredOutputParser,ResponseSchema


from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain import LLMChain

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import pprint

import warnings
warnings.filterwarnings("ignore")
import param
pn.extension('codeeditor')


# In[2]:


api_key='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' #Enter your api key here


# In[ ]:





# In[ ]:





# # chatbot

# In[3]:


class ChatGPT():
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = self.api_key
        self.llm = OpenAI()
        self.chat_model = ChatOpenAI()
#         print(self.llm)

        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=30, return_messages=True)

        
    def get_hints(self, question):
        final_schema = [
            ResponseSchema(name='Hint1', description='Give the first Hint'),
            ResponseSchema(name='Hint2', description='Give the Second Hint'),
            ResponseSchema(name='Hint3', description='Give the Third Hint'),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(final_schema)
        format_instructions = output_parser.get_format_instructions()

        # Creating a template
#         question_template = """Please give me hints to solve this question {question}. Dont provide me the code directly to
#         so that I can get my hands on programming.{instructions}"""
        question_template = """Please give direction to solve this question {question}. Dont provide me the code directly to
        so that I can get my hands on programming.{instructions}"""


        # Creating a Prompt
        question_prompt = PromptTemplate(template=question_template, input_variables=['question'], partial_variables={'instructions': format_instructions})

        # Creating a Chain
        Chain_one = LLMChain(prompt=question_prompt, llm=self.llm)

        # Making the chain and executing it
        answer = Chain_one.run({'question': question})

#         print(answer)

        # Storing it in memory to get the context
        self.memory.save_context({"input": question}, {"output": answer})
        
        self.question=question
        Q=self.question
        
                
        return answer

    def hint_explanation(self, hint_no):
        explanation_prompt="""Thanks for the Hints but I did not understand HINT{} please elaboate.""".format(hint_no)
        conversation_with_summary = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            verbose=False
        )
        explanation=conversation_with_summary.predict(input=explanation_prompt)
#         print(explanation)
        return explanation
        
    def evaluate_code(self,answer,ques):
#         evaluate_template="""I have written this code enclosed in backtics ``{answer}`` for {question}.
#                     Please evaluate it for accuracy ,readablity and optimaization each out of 5.Give me syntax errors if any."""

        evaluate_template="""I have written this code enclosed in backtics ``{answer}`` for {question}.
                    Evaluate this."""
        evaluate_prompt=PromptTemplate(template=evaluate_template,input_variables=['answer','question'])
        evaluate_Chain=LLMChain(prompt=evaluate_prompt,llm=self.llm)
        evaluation=evaluate_Chain.run({'question':ques,'answer':answer})
#         print(evaluation)
        return evaluation


class ChatBot():
    def __init__(self,api_key):
        self.api_key=api_key
        self.conversation=[]
        
        self.ai=ChatGPT(api_key)
        
    def ask(self,question=""):
        if len(question)!=0:
            self.conversation.append({"You":question})
            answer=self.ai.get_hints(question)
            self.conversation.append({"Ai Adventures":answer})
            
        chat_box = pn.widgets.ChatBox(
            value=self.conversation,
            allow_input =False,
            ascending=True
            )
        
        display(chat_box)
        return chat_box

    def get_explanation(self,hint_no):      
        explain="I want explanation for Hint"+str(hint_no)
        
        self.conversation1=[]
        
        self.conversation1.append({"You":explain})
        explanation=self.ai.hint_explanation(hint_no)
        self.conversation1.append({"Ai Adventures":explanation})
        
        chat_box = pn.widgets.ChatBox(
            value=self.conversation1,
            allow_input =False,
            ascending=True
            )

        display(chat_box)
    def create_code_editor(self):
        py_code = "#Type your code here"
        editor = pn.widgets.CodeEditor(value=py_code, sizing_mode='stretch_width', language='python', height=300,theme='cobalt')
#         display(editor)
        return editor
    
    def send_code_for_evaluation(self,code,Q):
        evaluation=self.ai.evaluate_code(code,Q)
        
        self.conversation2=[]
        
        self.conversation2.append({"You":"Please Evaluate My Code"})
        self.conversation2.append({"Ai Adventures":evaluation})
        chat_box = pn.widgets.ChatBox(
            value=self.conversation2,
            allow_input =False,
            ascending=True
            )
        display(chat_box)
        
class front_end(ChatBot):
    def __init__(self):
        self.model=ChatBot('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')
        self.textarea = widgets.Textarea(layout=widgets.Layout(width='800px'))
        self.button = widgets.Button(description='ASK', layout=widgets.Layout())
        self.button.style.button_color = '#76B900'
        self.button.style.font_color = '#FFFFFF'
        self.button.on_click(self.print_output)
#         display(widgets.VBox([self.textarea , self.button]))
        
        self.button3 = widgets.Button(description='Code Editor', button_style='info')

        self.button3.on_click(self.get_code)
        self.button3.style.button_color = '#0A66C2'
        self.button3.style.font_color = 'white'
        
        spacer = widgets.Box(layout=widgets.Layout(flex='1'))

        buttons_row = widgets.HBox([self.button, spacer, self.button3])
        
        container = widgets.VBox([self.textarea, buttons_row])
        display(container)
    
    def print_output(self,event):
        output=self.textarea.value        
        chat_box=self.model.ask(output)
        
        self.textarea2 = widgets.IntText(description='Enter the hint Number you want explanation for:', min=0, max=100)
        self.button2 = widgets.Button(description='Fetch Value', button_style='primary')
        self.button2.on_click(self.get_hint_no)

        display(widgets.VBox([self.textarea2, self.button2], layout=widgets.Layout(justify_content='center')))      
        
    def get_hint_no(self,event):
        hint_no = self.textarea2.value
        self.model.get_explanation(hint_no)
        
    def get_code(self,event):
        self.code_editor=self.model.create_code_editor()
        
                
        display(self.code_editor)
        self.button4=widgets.Button(description='Evaluate Code', button_style='success')
        self.button4.on_click(self.get_code_for_evaluation)
        
        self.question=self.textarea.value
#         display(self.question)
        
        display(self.button4)

    def get_code_for_evaluation(self,event):
        code=self.code_editor.value
#         print(code[21:])
        self.Q=self.textarea.value
        
        self.model.send_code_for_evaluation(code[21:],self.Q)
# pn.save('one.html')


# In[ ]:





# Write a Python function called most_frequent_word that takes in a string of text as input and returns the most frequent word in the text. 
# If there are multiple words with the same maximum frequency, return the one that appears first.

# In[4]:


# user=front_end()
# what is 1+1?
# Function to sort the dictionary by values
# give a function to get second largest element from the list


# def most_frequent_word(text):
#     word_count = {}
#     words = text.lower().split()
#     
#     for word in words:
#         word = word.strip('.,!?')
#         if word in word_count:
#             word_count[word] += 1
#         else:
#             word_count[word] = 1
#     
#     most_frequent = max(word_count, key=word_count.get)
#     return most_frequent
# 

# In[ ]:





# In[6]:


# '```json'.replace('```json','')


# In[1]:


# get_ipython().run_cell_magic('writefile', 's.py', '')


# In[ ]:





# In[ ]:




