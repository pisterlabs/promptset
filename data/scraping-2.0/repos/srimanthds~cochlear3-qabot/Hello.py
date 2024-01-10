#!/usr/bin/env python
# coding: utf-8

# In[15]:

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
import os
import shutil
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo
import joblib
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import pandas as pd



# In[2]:


PDF_FOLDER_PATH = "Data/"
LOADED_PDF_FILES_PICKLE = "loaded_pdf_files_pickle.pkl"
VECTOR_SEARCH_PICKLE = "vector_search_pickle.pkl"
DB_NAME = "cochlear_13"
COLLECTION_NAME = "vectorSearch"
INDEX_NAME = "default"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0


# In[3]:


def get_secret_key():
    open_api_key = st.secrets.open_api_key
    if not open_api_key:
        raise ValueError("The open_api_key environment variable is not set.")
    s1 = st.secrets.db_username
    s2 = st.secrets.db_pswd
    atlas_connection_string = "mongodb+srv://{s1}:{s2}@cluster0.1thtla4.mongodb.net/?retryWrites=true&w=majority".format(s1 = s1, s2 = s2)
    if not atlas_connection_string:
        raise ValueError("The atlas_connection_string environment variable is not set.")
    secret_key_dict = {"open_api_key": open_api_key, "atlas_connection_string": atlas_connection_string}
    return secret_key_dict


# In[4]:


def get_vector_search_object(cluster,db_name,collection_name, index_name,open_api_key):
    mongodb_collection = cluster[db_name][collection_name]
    # doc =  Document(page_content="dummy text", metadata={"source": "dummy"})
    # vector_search = MongoDBAtlasVectorSearch.from_documents(
    #                 documents=[doc],
    #                 embedding=OpenAIEmbeddings(api_key=open_api_key),
    #                 collection=mongodb_collection,
    #                 index_name=index_name 
    #             )
    embedding=OpenAIEmbeddings(api_key=open_api_key)
    vector_search = MongoDBAtlasVectorSearch(mongodb_collection, embedding)
    return vector_search


# In[5]:


def connect_mongodb(atlas_connection_string):
    cluster = MongoClient(atlas_connection_string)
    try:
        cluster.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return cluster


# In[17]:


# def get_prompt():
 
#   prompt_template="""
#     role='You are an expert acting as an helpful chatbot assistant who provides call center agents with accurate information retrieved from context without hallucinating'
#     instructions='1. You must start your response with Hi and Generate an accurate response according to the user question by referring to information provided in the context
#     2.Your response should not bring any external information apart from context i am sharing 3.If you dont have enough information to answer the question, Please respond that you dont have sufficient knowledge to answer the question'
#     details='response should give the information you think is correct based on the question and conclude your response with yes/no if required'
#     examples='''
#     'Q': "I am flying to Dubai tomorrow and its 60 degrees celsius there,  is it safe to travel there ?", "context": context provided in this prompt template,
#             "A":"Reasoning- In dubai current temperature is 60 degrees, According to source information Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C. According to source  the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors, Since 60 degrees in dubai is > 5 degrees and greater than 40 degrees, I would say exposing to extreme temperatures would need doctors recommendation. ANSWER- Hence say No, Not recommended ".
#   'Q': "I am flying to canada tomorrow and its -10 degrees celsius there,  is it okay to travel to canade with extreme low temperatures after my implant surgery ?",
#             "context": context provided in this prompt template,
#             "A":"Reasoning- In canada  temperature is -10 degrees, According to source information  Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C. According to source  the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors, Since -10 degrees temperature  in canada is < -5 and 40 degrees, I would say exposing to such low temperatures would need doctors recommendation. ANSWER-No, Not recommended ".
#   'Q': "  'Q': "I am flying to India tomorrow and its 45 degrees celsius there because of hot summer,  is it safe to travel there as i had implant surgery recently ?",
#             "context": context provided in this prompt template,
#             "A":"Reasoning- In India current temperature is 45 degrees,According to source information Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C." \
#             +"According to source  the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors,  Since 45 degrees in India is greater than the upper thresold 40 degrees and greater than 5 degrees of lower thresold for sound processors, I would say exposing to extreme temperatures would need doctors recommendation. ANSWER-No, Not recommended without medical advice".
#   'Q': "I am flying to saudi arabia next month and its expected teperature is 35 degrees celsius there,  is it safe to travel there ?",
#             "context": '''Extreme temperatures may be experience in some countries during seasonal periods or in a car parked in the sun.
# Extreme temperatures may also be experienced in e.g. saunas or medical treatment (cold chamber).The sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C.
# The implant incorporated in the body will not be exposed to extreme temperatures. Recommendation: The recipient can undergo extreme temperatures (e.g. sauna, cold chamber) without any harm to the implant.
# The externals should be taken off while undergoing this procedure. Recipients should follow the user manual in relation to storage of the external equipment and batteries
# (e.g. not to leave externals on a hot day on the dashboard of an automobile)''',
#             "A":"Reasoning- In saudi arabia if expected temperature for next month is 35 degrees, After validating with source information Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C." \
#             +" Since 35 degrees in saudi arabia is less than +40°C and greater than +5°C the temperature is falling within the thresold i.e.., +5°C to +40°C  for sound processors,It is safe to travel. ANSWER- YES".         
#   'Q': "I would like to do under water diving at a depth of 60 meters, will tthis harm my Nucleus CI24R device",
#             "context": '''The Nucleus CI24R, CI24M and CI22M implants are validated to withstand pressure at a depth of 25m under water for the purposes of scuba diving, which is equivalent to 2.5 atm nominal pressure and 4 atm test pressure.
# The Nucleus CI500 series and Freedom (CI24RE) implants are validated to withstand pressure at a depth of 40m under water for the purposes of scuba diving, which is equivalent to 4 atm nominal pressure and 6 atm test pressure.
# Recipients should seek medical advice before participating in a dive for conditions that might make diving contraindicated, e.g. middle ear infection, etc.
# When wearing a mask avoid pressure over the implant site''',
#             "A":"Reasoning- According to source information Sound processors are specified to withstand pressure at a depth of 40m under water for the purposes of scuba diving" \
#             +"you are willing to do diving to 60 meters for sound processors,since 60 meters >40 meters where 40 meters is the maximum withstandable pressure for this device as per the souce information. It is not recommended"
#             ANSWER- YES".'''
#   directions=''' "The response should match the information from context and no external data should be used for generating response",
#                 "call center agent question may contain numerical fields in it. If yes, then compare numeric values with thresold values available in context and validate it twice before giving response",
#                 "If you are not sure of answer, Acknowledge it instead of giving wrong response as misinformation may lead to loss of trust on you" ''' 
#   validation='Always validate your response with instructions provided.'
#   Context: {context}
#     Question: {question}  
#   """
 
#   prompt = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question","role","instructions","details","examples","directions","validation"]
#     )
  
#   return prompt

# def get_prompt():
 
#   prompt_template="""
#     role='You are an expert acting as an helpful chatbot assistant who provides call center agents with accurate information retrieved from context without hallucinating'
#     instructions='1. You must start your response with Hi and Generate an accurate response according to the user question by referring to information provided in the context
#     2.Your response should not bring any external information apart from context i am sharing 3.If you dont have enough information to answer the question, Please respond that you dont have sufficient knowledge to answer the question'
#     details='response should give the information you think is correct based on the question and conclude your response with yes/no if required'
#     examples='''
#   'Q': "I am flying to canada tomorrow and its -10 degrees celsius there,  is it okay to travel to canade with extreme low temperatures after my implant surgery ?",
#             "context": context provided in this prompt template,
#             "A":"In canada  temperature is -10 degrees, According to source information  Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C. According to source  the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors, Since -10 degrees temperature  in canada is < -5 and 40 degrees, 
#             I would say exposing to such low temperatures would need doctors recommendation.No,Not recommended".
#   'Q': "  'Q': "I am flying to India tomorrow and its 45 degrees celsius there because of hot summer,  is it safe to travel there as i had implant surgery recently ?",
#             "context": context provided in this prompt template,
#             "A":"In India current temperature is 45 degrees,According to source information Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C." \
#             +"According to source  the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors,  Since 45 degrees in India is greater than the upper thresold 40 degrees and greater than 5 degrees of lower thresold for sound processors, I would say exposing to extreme temperatures would need doctors recommendation.Not recommended without medical advice."
#   'Q': "I am flying to saudi arabia next month and its expected teperature is 35 degrees celsius there,  is it safe to travel there ?",
#             "context": '''Extreme temperatures may be experience in some countries during seasonal periods or in a car parked in the sun.
# Extreme temperatures may also be experienced in e.g. saunas or medical treatment (cold chamber).The sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C.
# The implant incorporated in the body will not be exposed to extreme temperatures. Recommendation: The recipient can undergo extreme temperatures (e.g. sauna, cold chamber) without any harm to the implant.
# The externals should be taken off while undergoing this procedure. Recipients should follow the user manual in relation to storage of the external equipment and batteries
# (e.g. not to leave externals on a hot day on the dashboard of an automobile)''',
#             "A":"In saudi arabia if expected temperature for next month is 35 degrees, After validating with source information Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C. Since 35 degrees in saudi arabia is less than +40°C and greater than +5°C the temperature is falling within the thresold i.e.., +5°C to +40°C  for sound processors.Yes, Its safe to travel".         
#   'Q': "I would like to do under water diving at a depth of 60 meters, will this harm my Nucleus CI24R device",
#             "context": '''The Nucleus CI24R, CI24M and CI22M implants are validated to withstand pressure at a depth of 25m under water for the purposes of scuba diving, which is equivalent to 2.5 atm nominal pressure and 4 atm test pressure.
# The Nucleus CI500 series and Freedom (CI24RE) implants are validated to withstand pressure at a depth of 40m under water for the purposes of scuba diving, which is equivalent to 4 atm nominal pressure and 6 atm test pressure.
# Recipients should seek medical advice before participating in a dive for conditions that might make diving contraindicated, e.g. middle ear infection, etc.
# When wearing a mask avoid pressure over the implant site''',
#             "A":"According to source information Sound processors are specified to withstand pressure at a depth of 40m under water for the purposes of scuba diving you are willing to do diving to 60 meters for sound processors,since 60 meters >40 meters where 40 meters is the maximum withstandable pressure for this device as per the souce information hence it is not recommended. Yes,it may harm the device".'''
#   directions=''' "The response should match the information from context and no external data should be used for generating response",
#                 "call center agent question may contain numerical fields in it. If yes, then compare numeric values with thresold values available in context and validate it twice before giving response",
#                 "If you are not sure of answer, Acknowledge it instead of giving wrong response as misinformation may lead to loss of trust on you" ''' 
#   validation='Always validate your response with instructions provided.'
#   Context: {context}
#     Question: {question}  
#   """
 
#   prompt = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question","role","instructions","details","examples","directions","validation"]
#     )
  
#   return prompt


def get_prompt():

  prompt_template="""
    role='You are an expert acting as an helpful chatbot assistant who provides call center agents with accurate information retrieved from context without hallucinating'
    instructions='1. You must start your response with Hi and Generate an accurate response according to the user question by referring to information provided in the context
    2.Your response should not bring any external information apart from context that is provided  
    3.If you dont have enough information to answer the question, Please respond that you dont have sufficient knowledge to answer the question

    details='response should give the information you think is correct based on the question and conclude your response accordingly'

    Following are the examples with "Q" referring to the Question. "Reasoning" reffers to the reasoning on how to derive the answer. "Answer" reffers to the final Answer.

    examples='''
  'Question': "I am flying to Dubai tomorrow and its 60 degrees celsius there,  is it safe to travel there wearing the sound processors ?"
  "Reasoning":  In dubai current temperature is 60 degrees, According to the context, Sound processors are specified for operating Temperatures between +5°C to +40°C and storage temperatures between -20°C to +50°C." \
            +" According to the context, the operating temperatures thresold i.e.., +5°C to +40°C  for sound processors, Since 60 degrees in dubai is > 5 degrees and greater than 40 degrees, I would say exposing to extreme temperatures would need doctors recommendation.

  "Answer"- "As the operating temperatures are between +5°C to +40°C, it is not recommended to travel there with the implant as the temperature is 60 degrees".
  
  'Question': "I would like to do under water diving at a depth of 60 meters, will tthis harm my Nucleus CI24R device",
  "Reasoning- According to the context Nucleus CI24R device are specified to withstand pressure at a depth of 40m under water for the purposes of scuba diving" \
            +"you are willing to do diving to 60 meters for sound processors,since 60 meters >40 meters where 40 meters is the maximum withstandable pressure for This device as per the souce information. It is not recommended"
  "Answer"- Yes, this will harm my device. As Nucleus CI24R device can withstand only upto the depths of 40m and since diving to 
  60m is above 40m. It will harm the device.
            '''
          
  directions='''"As per the above examples, you are supposed to understand the question, and based on the Context provided only, you must first reason out logically and accurately and respond back by adding the facts from the context and giving your response" 
                "The response should match the information from context and no external data should be used for generating response. Ensure you say you do not know if the answer to the question is not provided in the context",
                "call center agent question may contain numerical fields in it. If yes, then compare numeric values with thresold values available in context and validate it twice before giving response",
                "If you are not sure of answer, Acknowledge it instead of giving wrong response as misinformation may lead to loss of trust on you" '''
   validation='Always validate your response with instructions provided. Ensure you say you do not know if the answer is not provided in the Context'
  output= 'You need to respond back with the Answer without any prefixes such as "Answer:"'
  #Input
  Context: {context}
  Question: {question}
  
  #Ouput
  Answer statement
  """

  prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

  return prompt


def get_prompt_critique():
    prompt_template = """You are the smart engine that looks at the response below along with the question asked
    and makes edit to the response only if you think the response needs to be edited due to logical or contradicting mistakes

    1. First read the question stated below and understand it.
    2. Read the response below. This response acts as the answer to the question. However this response may be semantically
    or logically incorrect in response.
    3. The response usually will have 2 parts, the first part will be the answer and the second part will have the context 
    or information or reasoning from which the answer was stated.    
    4. If the answer and the reason are not in alignment, reformulate the response and send the correct response again
    5. If the original response doesn't have "Yes/No", do not forcefully add "Yes/No" in the beginning.

    Here are few examples for you to understand - 

    Question: I have Cochlear Implant series and want to swim to 30 meters, will this harm my device? 

    Response: No, the Cochlear Implant series are validated to withstand pressure up to 40m under water for the 
    purposes of swimming, which is equivalent to 4 atm nominal pressure and 6 atm test pressure. Therefore, swimming to 
    30 meters will not cause any harm to your device.
    
    Reformulated/Revised Response: No, the Cochlear Implant series are validated to withstand pressure up to 40m under water for the 
    purposes of swimming, which is equivalent to 4 atm nominal pressure and 6 atm test pressure. Therefore, swimming to 
    30 meters will not cause any harm to your device.
    
    Reason: In the Response, it clearly says that the device can withstand upto 40m and in the Question, the question asked is
    can it go to 30m and will it harm the device. Since it doesn't harm the device, the answer should be "No" followed by the 
    same text that's in Response. Hence this is not having contradicting response, hence the same Response has been replied back
    as Revised Response without changing anything
    
    Question: I have Cochlear Implant series and want to swim to 50 meters, will this harm my device? 

    Response: No, the Cochlear Implant series are not designed to withstand pressure at depths greater than 40m 
    for swimming. Therefore, swimming to a depth of 50m would exceed the recommended pressure and could cause damage 
    to the implant.
    
    Reformulated/Revised Response: Yes, the Cochlear Implant series are not designed to withstand pressure at depths greater than 
    40m for swimming. Therefore, swimming to a depth of 50m would exceed the recommended pressure and could cause damage 
    to the implant.
    
    Reason: The Question clearly asked if it will harm the device when a person goes swimming to 50m, the Response says that
    it will harm the device if it goes beyond 40m. But it has "No" and this is contradicting to the question asked. Hence
    "No" has been changed to "Yes" and the rest of the reason is never changed. The reason should never be changed and only the
    response such as "yes"/"no" can be changed based on the question asked.
    
    From the above 2 examples, understand the context of the question and understand the response and understand how the 
    revised response has been changed or kept the same throught the reason. The reason is for you to understand logically how
    you need to respond back.
    
    Remember, "Response" is the source truth and you need to only believe it and not bring any other external sources. You need
    to only change the "Yes/No" part of the question and not change anything else. This is very important
    
    
    Be precise and accurate and be logical in answering. 

    If the original response doesn't have "Yes/No", do not forcefully add "Yes/No" in the beginning.
    
    While formulating it be accurate and logical. Do not give contradicting answers. 

    The response should be the only facts you will look out for and not any other external
    facts. While formulating the response read the question again and answer accordingly to avoid contradicting replies

    Reply with the reformulated response.

    Just send the response, do not prefix with anything like "Response :" or "Revised Response :"

    Question: {Question}
    
    Response: {Response}
    
    Reformulated/Revised Response: Your Revised Response


    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["Question", "Response"]
    )
    return prompt

# In[20]:
def get_prompt_critique2():
    prompt_template = """You are the smart engine that looks at the response below along with the question asked and makes edit to the response only if you think the response needs to be edited due to logical or contradicting mistakes.If the response below says its not confident and doesn't have knowledge then mention the same as your response
    Question: {Question}
    Response: {Response}
    Reformulated/Revised Response: Your Revised Response
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["Question", "Response"]
    )
    return prompt

def get_response(db_name, collection_name, index_name, query):
    secret_key_dict = get_secret_key()
    open_api_key = secret_key_dict["open_api_key"]
    atlas_connection_string = secret_key_dict["atlas_connection_string"]
    cluster = connect_mongodb(atlas_connection_string)
    vector_search = get_vector_search_object(cluster,db_name,collection_name, index_name, open_api_key)
    qa_retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "post_filter_pipeline": [{"$limit": 25}]},
    )
    prompt = get_prompt()
    try:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=open_api_key,temperature=0),
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    except:
        time.sleep(120)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=open_api_key,temperature=0),
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    docs = qa({"query": query})

    # print(docs["result"])
    # print(docs["source_documents"])
    return docs


# In[ ]:

result = []
# Page title
st.set_page_config(page_title='Cochlear Smart QA Engine')
st.title('Cochlear Smart QA Engine')

# # File upload
# uploaded_file = st.file_uploader('Upload an article', type='pdf')
# print(dir(uploaded_file))
# Query text

secret_key_dict = get_secret_key()
open_api_key = secret_key_dict["open_api_key"]

if 'qa_data' not in st.session_state:
    st.session_state.qa_data = {'question': '', 'rag_responses': [], 'responses': []}

streamlit_pwd = st.secrets.streamlit_pwd
# Form input and query


user_input = st.text_input('Enter the application password:', type='password')
if user_input != streamlit_pwd:
    st.error("Authentication failed. Please provide the correct password.")
else:
    with st.form('myform', clear_on_submit=True):
        query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=False)

        # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
        submitted = st.form_submit_button('Submit')
        
        if submitted:
            with st.spinner('Calculating...'):
                try:
                    docs = get_response(DB_NAME,COLLECTION_NAME,INDEX_NAME,query_text)
                except:
                    time.sleep(120)
                    docs = get_response(DB_NAME,COLLECTION_NAME,INDEX_NAME,query_text)
                if (len(docs) != 0) and ("result" in dict(docs).keys()):

                    response = docs["result"]
                    rag_response = response
                    st.session_state.qa_data['rag_responses'].append(response)
                    try:
                        prompt = get_prompt_critique2()
                        llm = OpenAI(api_key=open_api_key,temperature=0)
                        prompt.format(Question=query_text,Response=response)
                        chain1 = LLMChain(llm=llm,prompt=prompt)
                        response = chain1.run(Question=query_text,Response=response)
                    except:
                        time.sleep(120)
                        prompt = get_prompt_critique2()
                        llm = OpenAI(api_key=open_api_key,temperature=0)
                        prompt.format(Question=query_text,Response=response)
                        chain1 = LLMChain(llm=llm,prompt=prompt)
                        response = chain1.run(Question=query_text,Response=response)
                        
                    result.append(response)
                    st.session_state.qa_data['question'] = query_text
                    st.session_state.qa_data['responses'].append(response)
                    for idx, r in enumerate(st.session_state.qa_data['responses'][::-1], start=1):

                        # Split the response into words
                        words = rag_response.split(' ')                    
                        # Initialize an empty line and list of lines
                        line, lines = '', []
                        
                        # Add words to the line until it exceeds the desired width
                        for word in words:
                            if len(line + word) > 10:
                                lines.append(line)
                                line = word + ' '
                            else:
                                line += word + ' '
                        
                        # Add the last line
                        lines.append(line)
                        
                        # Join the lines with newline characters
                        formatted_response = '\n'.join(lines)
                        
                        # Display the formatted response
                        st.info(f"Question: {query_text} \n\n {formatted_response} \n\n")
                        



                        # st.info(f"Question: {query_text} \n\n {rag_response} \n\n")
                        # st.markdown(f"""**Question:** {query_text}\n {rag_response}""")
                        # st.info(f"Question: {query_text} \n\n {rag_response} \n\n")
                        #st.info(f"Question: {query_text} \n\n {rag_response} \n\n Response : {r} \n\n")
                        
                        # st.info(f"RAG Response : {rag_response}")
                        # st.info(f"Response : {r}")

                    st.title('Top Similar Documents')
                    df_lis = []
                    for i in docs["source_documents"]:
                        lis = []
                        lis.append(i.page_content)
                        if "source" in i.metadata.keys():
                            lis.append(i.metadata["source"])
                        else:
                            lis.append("")
                        if "page" in i.metadata.keys():
                            lis.append(i.metadata["page"])
                        else:
                            lis.append(None)
                        df_lis.append(lis)
                    similar_df = pd.DataFrame(df_lis,columns = ["Text", "Source Document", "Page Number"])

                    st.table(similar_df)
                
                else:
                    st.session_state.qa_data['question'] = query_text
                    st.session_state.qa_data['responses'] = None
    #             del openai_api_key
    st.write(f"Last Submitted Question: {st.session_state.qa_data['question']}")
    st.write("All Responses:")
    for idx, r in enumerate(st.session_state.qa_data['rag_responses'], start=1):
        st.write(f"RAG Response : {r}")
    for idx, r in enumerate(st.session_state.qa_data['responses'], start=1):
        st.write(f"Response {idx}: {r}")
        # if len(result):
        #     st.info(response)

