import streamlit as st
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
# from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
# from langchain.chains.summarize import load_summarize_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO
import os, webvtt, json
# import openai
import tiktoken


# Set the API key for the openai package
os.environ['OPENAI_API_KEY']  = st.secrets['OPEN_AI_KEY']
chunk_size = 3000
model = 'gpt-3.5-turbo'
chat = ChatOpenAI(model_name=model,temperature=0)
llm_predictor = LLMPredictor(llm=chat)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

st.set_page_config(page_title="Teams-Turner",page_icon="⚡")

st.title("⚡ Teams-Turner")
st.write('Like the "Time-Turner" from Harry Potter 🧙‍♀️')
st.write('to be in two Teams meetings at the same time')


# load document with LLaMa Index
def load_doc():
    documents = SimpleDirectoryReader('txt').load_data()
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    query_engine = index.as_query_engine()
    return query_engine


def num_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def slice_string(text: str) -> list[str]:
    # Split text into chunks based on space or newline
    chunks = text.split()

    # Initialize variables
    result = []
    current_chunk = ""

    # Concatenate chunks until the total length is less than 4096 tokens
    for chunk in chunks:
        # if len(current_chunk) + len(chunk) < 4096:
        if num_tokens(current_chunk+chunk) < chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            result.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        result.append(current_chunk.strip())

    return result

def summarize(context: str, convo: str) -> str:
    """Returns the summary of a text string."""
    res = chat([SystemMessage(content=context),
        HumanMessage(content=convo)])
    return res.dict()['content']

def refine(summary: str,context: str, chunk: str) -> str:
    """Refine the summary with each new chunk of text"""
    context = "Refine the summary with the following context: " + summary
    summary = summarize(context,chunk)
    return summary

context = 'summarize the following conversation'
file = st.file_uploader('Upload Teams VTT transcript',type='vtt')
maxtokens = {'gpt-3.5-turbo': 4096,'gpt-4': 8192 }

if file is not None:
    data = StringIO(file.getvalue().decode('utf-8'))
    disc = webvtt.read_buffer(data)
    part = st.checkbox('include participants')
    time = st.checkbox('include time')
    str = []
    for caption in disc:
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

    if st.button('summarize'):

        if (toknum > maxtokens[model]-1000):
            # st.write(f'Text too long please prune to fit under {maxtokens[model]-1000} tokens')
            chunks = slice_string(convo)
            summary = summarize(context,chunks[0])
            for chunk in chunks[1:]:
                summary = refine(summary,context,chunk)
            st.write(summary)
        else:
            st.write(summarize(context,convo))


else:
    with open('vtt/YannMike_2023-03-08.vtt') as f:
        st.download_button(
            label="Sample VTT file",
            data=f,
            file_name="sample.vtt",
            mime="text/vtt"
          )

if st.checkbox('Ask questions about the meeting'):
    query_engine = load_doc()
    # st.write(index)
    q1 = 'How does Yann want to develop an app?'
    query = st.text_input('Query',q1)
    if st.button('Answer'):
        res = query_engine.query(query)
        st.write(res.response)
        with st.expander('Sources:'):
            st.write(res.source_nodes[0].node.get_text())