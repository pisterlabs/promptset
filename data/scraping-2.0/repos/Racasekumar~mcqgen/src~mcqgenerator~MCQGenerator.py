import os
import json
import PyPDF2
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging


#importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback



#load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environment
key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo",temperature=0.7)

with open("/config/workspace/Response.json","r") as f:
    RESPONSE_JSON=json.load(f)
    
print(RESPONSE_JSON)

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template
    )

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)


template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity.
If the quiz is not at per with the congnitive and analytical abilities of the student,\
update the quiz questions which needs to be changed and change the tine such that it perfectly fits the student ability.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
  input_variables=["subject","quiz"],
  template = template2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain,review_chain],
                                          input_variables=["text","number","subject","tone","response_json"],
                                          output_variables=["quiz","review"],
                                          verbose=True)
