from PyPDF2 import PdfReader
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
import  os
import streamlit as st
import docx

def get_data_from_resume_text(text)->pd.DataFrame:
  class Applicant(BaseModel):
    Applicant_Name : str = Field(description="Name of the Applicant/Candidate,if you cant find it return 'Not Available'")
    Mail_Id : str = Field(description="Mail ID of the Applicant/Candidate, if you cant find it return 'Not Available' ")
    years_of_exp : int = Field(description="Experience  of the Applicant/Candidate in years , if you cant find it return 'Not Available' ")
    Key_Skills : list = Field(descripion= "Top 10 Technical Skills of the Candidate eg. ['skill1','Skill2','skill3','skill4'] ")
    Linkedin_Profile : str = Field(description="Linkedin Profile link  of the Applicant/Candidate  , if you cant find it return 'Not Available' ")
    GitHub_Profile : str = Field(description="GitHub Profile link  of the Applicant/Candidate  , if you cant find it return 'Not Available' ")
    # Summary : str = Field(description = "Summarize the provided text in 100 words, the summary should have his work experience , key skills and years of experience ")

  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=Applicant)

  # Creating a Prompt from a prompt Template
  prompt = PromptTemplate(
      template="Extract the asked features only from the text.\n{format_instructions}\n{text}\n",
      input_variables=["text"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  llm = OpenAI(temperature=0)
  # using LCEL to create a pipeline which has prompt and llm
  extractor = prompt|llm
  # parsing the output to get
  output = parser.parse(extractor.invoke({"text":text})).dict()

  llm = OpenAI(temperature=0)
  output2 = llm(f"""Summarize the text delimited by three dashes in 100 words. dont include his/her hobbies/interest
--- 
 {text}
---          
 """)
  output['Summary']=output2.replace("\n"," ")
  return pd.DataFrame.from_dict(output,orient='index').T

@st.cache_data
def get_feature_df(target_dir: str)->pd.DataFrame:
    list_of_dfs = []
    for file in os.listdir(target_dir):
        text = ""
        if file.endswith(".pdf"):
            reader = PdfReader(target_dir + '/' + file)
            for page in reader.pages:
                text += page.extract_text()

        if file.endswith(".docx"):
            reader = docx.Document(target_dir + '/' + file)
            for page in reader.paragraphs:
                text += page.text

        list_of_dfs.append(get_data_from_resume_text(text))
    print("Read all the files and created meta data for all the resumes")
    df =  pd.concat(list_of_dfs, axis=0)
    df.Key_Skills = df.Key_Skills.astype(str)
    return df
@st.cache_resource
def get_vector_store(df:pd.DataFrame,vector_db_dir:str):
    docs = DataFrameLoader(df, page_content_column="Summary").load()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings,persist_directory=vector_db_dir)
    vectorstore.persist()
    print("Successfully Created Vector Store")
    return vectorstore

@st.cache_data
def get_matching_profiles(job_description: str, vector_db_dir:str) -> pd.DataFrame:
    list_of_dfs = []
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma( persist_directory=vector_db_dir,embedding_function=embeddings)
    matching_profiles = vectorstore.as_retriever().invoke(job_description)
    print("Successfully generated Matching Profiles")
    for profile in matching_profiles:
        list_of_dfs.append(pd.DataFrame.from_dict(profile.metadata, orient='index').T)

    return pd.concat(list_of_dfs, axis=0)