from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage)
from langchain.prompts.chat import (
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  )
import os

from dotenv import load_dotenv

load_dotenv()


BASE_URL = os.environ["BASE_URL"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
API_KEY = os.environ["API_KEY"]

model1 = AzureChatOpenAI(
openai_api_base=BASE_URL,
openai_api_version="2023-05-15",
deployment_name=DEPLOYMENT_NAME,
openai_api_key=API_KEY,
openai_api_type="azure",
temperature = 0)


llm = model1

def prompt_categorizer(text:str|None=None, set_of_tasks = set(["Summarization of given Information", "Creative Writing", "Interpret or Write Code",
                                                         "Natural Language Question"]) )->str:
 
 string_list = ""
 
 for i in list(set_of_tasks):
   string_list += i + "\n"
   
 Sys_Temp = "The User is prompting an Artificial Inteligence Model in a variety of tasks that need to be classified as: {types} \
 First, you will decompose the prompt into small specific and concrete tasks, in the following format: \
 To accomplish this prompt one needs to do the following tasks: \n \
Task 1: <Summary of task 1> ( <Type of Task 1> ) \n \
... \n \
Task N: <Summary of task N> ( <Type of Task N )"
 
 
 
 System_Prompt_template = SystemMessagePromptTemplate.from_template(template = Sys_Temp, )
 initial_prompt = System_Prompt_template.format_messages(types = string_list)[0]
 messages = [initial_prompt, HumanMessage(content = text)]
 result = llm(messages)#.content
 messages.append(result)
 
 formatting_prompt = "Good, now take each task presented and from that write a python list in the following form: \
 [<Type of task 1>, <Type of task 2>, ... ,<Type of task N>] \
 Be sure to respect each element of punctuation and formatting."
 
 Get_Tasks = "Good, now take each task presented and from that write a list in the following form: \
 [<Description of Task 1>; <Description of Task 2>; ... ;<Description of Task N>] \
 Be sure to separate each task by the character \";\" . Do not include the type of the task in this list."
 
 messages_class = messages +A[( HumanMessage(content = formatting_prompt) )]
 messages_tasks = messages + [HumanMessage(content = Get_Tasks)]
 
 result_class = llm(messages_class)
 result_class = result_class.content.replace("\"","").replace("\'","").strip("[]").split(", ")
 
 result_tasks = llm(messages_tasks)
 result_tasks = result_tasks.content.replace("\"","").replace("\'","").replace("\n","").strip("[]").split("; ")

 return zip(result_tasks,result_class)

if "__name__" == "__main__":
  AA = prompt_categorizer("Please explain to me the plot of the movie Cars.")
  dict(AA)
