from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
import pandas as pd
import json
from sqlalchemy import create_engine

import os
from dotenv import load_dotenv



load_dotenv(dotenv_path="../../.env")


POSTGRES_URL = os.getenv("POSTGRES_URL")



def ingestDataFromCSV(topic, subtopic, postgres_url):
    topic_list = pd.read_csv(f"scriptfiles/{topic}_list.csv", index_col="Index")

    engine = create_engine(postgres_url)

    for i, row in topic_list.iterrows():
        topic_item = row[topic]
        try:
            df = pd.read_csv(f"scriptfiles/{topic}/{subtopic}/{topic_item}_{subtopic}_content.csv", header=0)
            df = df.drop_duplicates()
            df['topic'] = topic_item
            df.to_sql(f"{topic}_{subtopic}", engine, if_exists='append', index=False)
        except:
            FileNotFoundError
            print(f"File not found for {topic_item}")
            continue
        
ingestDataFromCSV("sports","influencers", POSTGRES_URL)