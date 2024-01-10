#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import tempfile
import subprocess
import sys






# Step 1. Define Panel widgets

# In[1]:


import panel as pn 
pn.extension()

json_input = pn.widgets.FileInput(accept=".json", value="", height=50)
key_input = pn.widgets.PasswordInput(
    name="OpenAI Key",
    placeholder="sk-...",
)
k_slider = pn.widgets.IntSlider(
    name="Number of Relevant Chunks", start=1, end=5, step=1, value=2
)
chain_select = pn.widgets.RadioButtonGroup(
    name="Chain Type", options=["stuff", "map_reduce", "refine", "map_rerank"]
)
chat_input = pn.widgets.TextInput(placeholder="First, upload a json data!")


# In[ ]:


companies = []

def load():
    docs=[]
    for company in companies:
        text = ""
        text += "cik: " + company['cik'] + "\n"
        text += f"entityType: {company['entityType']}\n"
        text += f"sicDescription: {company['sicDescription']}\n"

        # Handle the 'tickers' field, which is an array
        tickers = ', '.join(company['tickers'])
        text += f"tickers: {tickers}\n"
        
        # Handle the 'exchanges' field, which is also an array
        exchanges = ', '.join(company['exchanges'])
        text += f"exchanges: {exchanges}\n"
        text += f"ein: {company['ein']}\n"
        text += f"category: {company['category']}\n"
        text += f"stateOfIncorporation: {company['stateOfIncorporation']}\n"
        text += f"fiscalYearEnd: {company['fiscalYearEnd']}\n"
        
        metadata = dict(
            #source=company['id'],
            name=company['name']
        )
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)
    return docs

#data = load()


# Step 2: Wrap LangChain Logic into a Function

# In[ ]:


def initialize_chain():
    if key_input.value:
        os.environ["OPENAI_API_KEY"] = key_input.value

    selections = (json_input.value, k_slider.value, chain_select.value)
    if selections in pn.state.cache:
        return pn.state.cache[selections]

    chat_input.placeholder = "Ask questions here!"

    # load document
    with tempfile.NamedTemporaryFile("wb", delete=False) as f: 
        f.write(json_input.value)
        
    file_name = f.name
    loader = load()
    documents = loader(file_name)

    
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": k_slider.value}
    )
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type=chain_select.value,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )
    return qa


# Step 3. Create a chat interface

# In[ ]:


async def respond(contents, user, chat_interface):
    if not json_input.value:
        chat_interface.send(
            {"user": "System", "value": "Please first upload a JSON!"}, respond=False
        )
        return
    elif chat_interface.active == 0:
        chat_interface.active = 1
        chat_interface.active_widget.placeholder = "Ask questions here!"
        yield {"user": "OpenAI", "value": "Let's chat about the JSON!"}
        return

    qa = initialize_chain()
    response = qa({"query": contents})
    answers = pn.Column(response["result"])
    answers.append(pn.layout.Divider())
    for doc in response["source_documents"][::-1]:
        answers.append(f"**Page {doc.metadata['page']}**:")
        answers.append(f"```\n{doc.page_content}\n```")
    yield {"user": "OpenAI", "value": answers}

chat_interface = pn.chat.ChatInterface(
    callback=respond, sizing_mode="stretch_width", widgets=[json_input, chat_input]
)
chat_interface.send(
    {"user": "System", "value": "Please first upload a JSON and click send!"},
    respond=False,
)


# Step 4. Customize the look with a template

# In[ ]:


template = pn.template.BootstrapTemplate(
    sidebar=[key_input, k_slider, chain_select], main=[chat_interface]
)
template.servable()

