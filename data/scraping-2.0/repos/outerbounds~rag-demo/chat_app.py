from llama_index import Document, VectorStoreIndex, set_global_service_context, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.llms import OpenAI
from dataclasses import dataclass
from metaflow import Flow
import streamlit as st
from typing import List
import time
import os

st.set_page_config(layout="centered")
st.title("Metaflow Chat Bot")
st.markdown("This is a bare bones LLM-powered chat bot that uses the results of Metaflow workflows to answer questions about Metaflow.")

subquery_prompt = """
    Answer this question only if there is relevant context below: {}
    If there is nothing in the context say: "Could not find relevant context."
    Here is the retrieved context: {}
"""

# model = st.text_input('OpenAI model', 'gpt-3.5-turbo')
# temp = st.slider(label='Temperature', min_value=0.0, max_value=1.0, step=0.01, value=0.0)
# chat_mode = st.text_input('LlamaIndex chat engine mode', 'react')
# K = st.number_input('K results to return', min_value=1, max_value=5, value=2, step=1)
model = 'gpt-3.5-turbo'
temp = 0.0
chat_mode = 'react'
K = 2

llm = OpenAI(model=model, temperature=temp, max_tokens=2048)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

# find latest Metaflow run that saved processed df
run = None
for _run in Flow('DataTableProcessor'):
    if _run.data.save_processed_df:
        run = _run
        break
st.write("Found processed df in run: {}".format(run.id))

@dataclass
class Context:

    def __init__(self, response: str, source_node_ids: List[str]):
        self.response = response
        self.source_node_ids = source_node_ids

    def get_link_df(self, meta_df, link_col = 'doc_id'):
        return meta_df[meta_df[link_col].isin(self.source_node_ids)]

def qa_iter(
    question: str, 
    index: VectorStoreIndex, 
    k:int = 2, 
    response_mode:str = 'tree_summarize'
) -> Context:
    "Match a question against an index and returns the response."
    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, retriever=retriever)
    query_res = query_engine.query(question)
    return Context(
        response=query_res.response, source_node_ids=list(query_res.metadata.keys())
    )

def st_output_format(similar_chunk_df):
    md_outputs = ["#### You may find these links helpful:"]
    for _, chunk in similar_chunk_df.iterrows():
        md_outputs.append(f"##### [{chunk.header}]({chunk.page_url})")
        md_outputs.append(f"{chunk.contents[:100]}...")
    if len(md_outputs) == 1:
        md_outputs = []
    return md_outputs

def generative_search_engine_iter(question, index, meta_df, k=2, meta_df_id_col='doc_id'):
    "Assumes index and df are defined in the global scope"
    context = qa_iter(question, index, k=k)
    similar_chunk_df = meta_df[meta_df[meta_df_id_col].isin(context.source_node_ids)]
    return context.response, st_output_format(similar_chunk_df)

# use the processed df to build the index
def get_documents_from_content_section_df(df):
    ids = []; documents = []
    for i, text in enumerate(df.contents):
        doc = Document(text=text, id_=i)
        documents.append(doc)
        ids.append(doc.id_)
    return documents, ids

@st.cache_resource
def setup_index():
    df = run.data.processed_df
    documents, ids = get_documents_from_content_section_df(df)
    df['doc_id'] = ids
    index = VectorStoreIndex(documents)
    return index, df

index, df = setup_index()
chat_engine = index.as_chat_engine(chat_mode=chat_mode, verbose=True, streaming=True)

# Initialize chat history
st.markdown("# Chat history")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_prompt := st.chat_input("Hey Metaflower 🌻 what's on your mind?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Finding context..."):
        subquery_response, md_outputs = generative_search_engine_iter(user_prompt, index, df, K)

    # Display assistant response in chat message container
    with st.chat_message("Metaflow assistant"):

        message_placeholder = st.empty()

        streaming_response = chat_engine.stream_chat(subquery_prompt.format(user_prompt, subquery_response))
        full_response = ""
        for text in streaming_response.response_gen:
            full_response += text
            message_placeholder.markdown(full_response + "▌")

        for line in md_outputs:
            st.markdown(line)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "Metaflow assistant", "content": full_response})

if st.button("Reset chat engine's memory"):
    chat_engine.reset()