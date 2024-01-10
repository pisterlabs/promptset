from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import openai
import os
import json
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import ast

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

#analyze the new task
#Given the ingredients json ocr, generate a usable summary
#Use both summaries stuffed in to prompt and create a smart result
def analyze(ehr, ingredients):
    ingredients = ast.literal_eval(ingredients)
    smart_prompt = f'''
    You are given the following brand and name details:
    {ingredients['metadata']}
    and the ingredients list of
    {ingredients['ingredients_list']}.

    Can you format this information and combine it into one standardized format?
    Do not redact or summarize any information regarding the ingredients list or dosage. Be as comprehensive as possible. Stick to formatting.'''

    classification = LLMChain(llm=OpenAI(temperature=0), prompt=PromptTemplate.from_template(smart_prompt))
    summary = classification.predict()

    query = f'''
    Act as a medical assistant that makes recommendations about if someone should take a medication or not given their doctor based diagnoses and 
    the details of the medication they plan to take.

    Patient's health record summary & highlights:
    {ehr}''' + '''

    Medication Brand, Name, and Ingredients:
    {ingredients_summary}
    '''

    prompt = ChatPromptTemplate.from_template(query)
    model = ChatOpenAI()
    chain = prompt | model

    result = chain.invoke({"ingredients_summary": summary})
    return summary, result