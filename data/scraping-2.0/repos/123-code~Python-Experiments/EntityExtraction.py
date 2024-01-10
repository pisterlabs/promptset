from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
import os
from neo4j import GraphDatabase
import glob 
from pydantic import BaseModel


openai_api_key=os.getenv("OPENAI_API_KEY")
schema = {
    "properties":{
        "skills":{"type":"string"},
    }
}
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
chain = create_tagging_chain(schema, llm)



def extract_entities(folder):
    files = glob.glob(f'data/{folder}/*')
    system_msg = "You are a helpful IT-project and account management expert who extracts information from documents."
    print(len(files))
    results = []


    for file in files:
        with open(file) as f:
            text = f.read()
            result = chain.run(text)
            results.append(result)
    return results

print(extract_entities('people_profiles'))
