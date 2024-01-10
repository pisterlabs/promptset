
import pandas as pd
import openai
import concurrent.futures

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-0613")

schema = {
    "properties" : {
        "evaluation" : {
            "type" : "integer",
            "enum" : [0,1]
        },
    },
    "required" : ["evaluation"]
}

chain = create_tagging_chain(schema, llm)

def eval_response(response):
  prompt = f"""The following is the response from a education chatbot which is trained to not respond on movie related questions. 

            Respond with 1 if the bot declines to respond only about movies or movie-related topics.  Else respond with 0. 

            <Bot response: {response}>
            """
  return chain.run(prompt)['evaluation']


# Read the Excel file
df = pd.read_excel('llel_check.xlsx') 

# Run the prompts concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    df['Eval'] = list(executor.map(eval_response, df['Responses']))

# Write the DataFrame with the responses back to the Excel file
df.to_excel('file_res.xlsx', index=False) 
