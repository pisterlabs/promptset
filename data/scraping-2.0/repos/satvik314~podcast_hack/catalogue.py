from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain
from langchain.prompts import ChatPromptTemplate
from supabase import create_client, Client
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
import os
from dotenv import load_dotenv
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

schema = {
    "properties" : {
        "category" : {"type" : "string", "enum" : ["Health and Wellness", "Productivity", "Technology", "Business", "Education"]},
    },
    "required" : ["category"]
}

tag_chain = create_tagging_chain(schema, llm)

def create_tag(title):
    return tag_chain.run(title)['category']

def get_catalogue():
    data = supabase.table('podcasts_db').select('created_at, category, topic, language, transcript').execute()

    df = pd.DataFrame(data.data)
    df = df.iloc[::-1]
    df = df.rename(columns={"transcript": "brief summary"})
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_at"] = df["created_at"].dt.strftime("%dth %B, %Y")
    df = df.rename(columns={"created_at": "listened_on"})
    df = df.reset_index(drop=True)  
    df.index = df.index + 1

    # st.dataframe(df)
    AgGrid(df)


