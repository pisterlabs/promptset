
#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import openai
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from openai import ChatCompletion


# In[9]:


# Set up Azure OpenAI
openai.api_type = "azure"
openai.api_base = "https://ausopenai.azure-api.net" # Api base is the 'Endpoint' which can be found in Azure Portal where Azure OpenAI is created. It looks like https://xxxxxx.openai.azure.com/
openai.api_version = "2023-07-01-preview"
openai.api_key = "6693a6eec2eb4b9b9f4ff83d5809fb36"


# In[10]:


system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""


# In[11]:


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


# In[12]:


def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')
    button_style = """
    position: absolute;
    top: 20px;
    right: 20px;
    background-color: #f0f0f0;
    padding: 10px;
    border: none;
    cursor: pointer;
    border-radius: 5px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    """
    st.markdown(
    f"""
    <style>
        .dropdown-button {{
            {button_style}
        }}
    </style>
    """,
    unsafe_allow_html=True,
    )
    Language = st.selectbox("Select Language", ["English", "Telugu", "French", "Spanish", "Hindi"])
  
    url = st.text_input("Insert The website URL")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        response = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k",
                messages=[{ "role": "system", "content": url + "\n" + prompt + " Give output in "+ Language}, { "role": "user", "content": "" }
                            ]
                )
        st.write(response['choices'][0]['message']['content'])
        

if __name__ == '__main__':
    main()


# In[14]:




