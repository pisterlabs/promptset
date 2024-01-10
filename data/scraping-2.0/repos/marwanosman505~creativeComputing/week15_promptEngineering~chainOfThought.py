from dotenv import load_dotenv
import os
import sys

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


manual = open(os.path.join(sys.path[0], "./data.json"), "r").read()
print(manual)
manual = manual.replace("{", "[")
manual = manual.replace("}", "]")

prompt1 = ChatPromptTemplate.from_template("You are a project delivery team lead and want to achieve {credits} out of the 4 credits in the manual:\n"+manual+"\n\nReturn the pros and cons for each credit and nothing else:")
prompt2 = ChatPromptTemplate.from_template(
    "Here are the pros and cons for each credit:{pros_and_cons}Return the credits that you will achieve and why:"
)

# get api key from environment variable
load_dotenv()
API_KEY = os.getenv("API_KEY")

# initialize model
model = ChatOpenAI(api_key=API_KEY, model="gpt-4")


chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"pros_and_cons": chain1} 
    |prompt2
    | model
    | StrOutputParser()
)
print(API_KEY)
print(chain2.invoke({"credits": "3"}))

    

