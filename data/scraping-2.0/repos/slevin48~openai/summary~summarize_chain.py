import streamlit as st
import webvtt
import os
# import openai
from io import StringIO
import tiktoken
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Summarize",page_icon="📝")

st.title('📝 Teams meeting summarizer (powered by 🦜🔗LangChain)')

# Set the API key for the openai package
os.environ['OPENAI_API_KEY'] = st.secrets['OPEN_AI_KEY']

def num_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize(model,doc):
    """Returns the summary of a text string."""
    llm = ChatOpenAI(model=model)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(doc)


model = st.radio('Model',('gpt-3.5-turbo','gpt-4'))
file = st.file_uploader('Upload Teams VTT transcript',type='vtt')
maxtokens = {'gpt-3.5-turbo': 4096,'gpt-4': 8192 }

if file is not None:
    data = StringIO(file.getvalue().decode('utf-8'))
    chat = webvtt.read_buffer(data)
    # data = file.getvalue().decode('utf-8')
    # with open('vtt/'+file.name,'w') as f:
    #     f.write(data)
    # caption = webvtt.read('vtt/'+file.name)
    part = st.checkbox('include participants')
    time = st.checkbox('include time')
    str = []
    for caption in chat:
        if part & time:
            str.append(f'{caption.start} --> {caption.end}')
            str.append(caption.raw_text)
        elif time:
            str.append(f'{caption.start} --> {caption.end}')
            str.append(caption.text)
        elif part:
            str.append(caption.raw_text)
        else:
            str.append(caption.text)
    sep = '\n'
    convo = sep.join(str)

    # Write to txt file
    with open('txt/chat.txt',mode='w') as f:
      f.write(convo)
        
    convo = st.text_area('vtt file content',convo)
    toknum = num_tokens(convo)
    st.write(toknum,'tokens')


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    loader = UnstructuredFileLoader("txt/chat.txt")
    doc = loader.load()

    
    if (toknum > maxtokens[model]-1000):
        doc = text_splitter.split_documents(doc)
    
    # st.write(doc)
    
    if st.button('summarize'):
        st.write(summarize(model,doc))

else:
    with open('vtt/YannMike_2023-03-08.vtt') as f:
        st.download_button(
            label="Sample VTT file",
            data=f,
            file_name="sample.vtt",
            mime="text/vtt"
          )
