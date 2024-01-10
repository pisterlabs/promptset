# LangChain & GPT 4 For Data Analysis: The Pandas Dataframe Agent
# https://www.youtube.com/watch?v=rFQ5Kmkd4jc

from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent  
import openai
import os

import logging  # Setup code for error logging                                        
logging.basicConfig(level=logging.DEBUG, format ='%(asctime)s - %(levelname)s - %(message)s') # basic logging format
logging.disable(logging.CRITICAL) # disables logging at all levels

load_dotenv(find_dotenv())  
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ignore or replace invalid characters: If you want to use 'utf-8', but you want to handle invalid characters gracefully, 
# you can use the 'ignore' or 'replace' error handlers. 'ignore' will simply remove invalid characters, while 'replace' will 
# replace them with the Unicode replacement character (�).

with open(r"C:\Users\dbigman\github\Projects\LangChain_data_analysis\gasco_sales_data.csv", 'r', encoding='utf-8', errors='ignore') as f:
    contents = f.read()

from io import StringIO
data = StringIO(contents)
df = pd.read_csv(data)

# logging.debug('df: (%s)' % (df))

chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
agent = create_pandas_dataframe_agent(chat, df, verbose = True)

while True:
    # Ask for a question from the user
    question = input("Please enter your question (or type 'QUIT' to exit): ")

    # Check if the user wants to quit
    if question.upper() == 'QUIT':
        break

    response = agent.run(question)
    print(response)
