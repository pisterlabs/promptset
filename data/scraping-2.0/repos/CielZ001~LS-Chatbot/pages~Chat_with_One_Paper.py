import streamlit as st
import pandas as pd
import pinecone
import os
import re

from langchain import PromptTemplate
from langchain.schema import ChatMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
xlsx_file_path = os.path.join(current_dir, 'r', 'merged_data02.xlsx')
txt_path =  os.path.join(current_dir, 'r', '1.txt')

# Logic of this code:
#
# 1. User input a title or topic
# 2. Search for the paper in title and abstract, to return a list of papers titles.
#    return the all matched results, remove duplicates, and display the list of titles in a selectbox
# 3. User select a paper title
# 4. Use the selected title to search for the paper in all-paper CSV file, return the paper's author, uri, title
#    and whole text to a list.
# 5. write the text into a text file
# 6. User ask a specific question of the paper content, use the text file as retriever of RetrievalQA chain,
#    to return the answer to the user's question.

# Set up initial configurations
OPENAI_API_KEY = st.secrets['openai-api-key']
pc_api_key = st.secrets['pc-api-key']
pc_env = st.secrets['pc-env']
pc_index = st.secrets['pc-index']

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# PINECONE_API_KEY = os.environ['pc_api_key']
# PINECONE_ENVIRONMENT = os.environ['pc_env']
# index_name = os.environ['pc_index']


class DocumentInput(BaseModel):
    question: str = Field()


def remove_duplicates(input_list):
    return list(set(input_list))

def row_to_dict(df, title):
    row_data = df.loc[df['title'] == title]
    return row_data.to_dict(orient='records')[0] if not row_data.empty else None

def search_excel(user_input):
    df = pd.read_excel(xlsx_file_path)
    search_cols = ['title', 'dc.abstract[en_US]']

    # apply case-insensitive search and get boolean dataframe
    mask = df[search_cols].apply(lambda x: x.str.contains(user_input, case=False)).any(axis=1)

    # filter dataframe by mask and get titles list
    titles_list = df.loc[mask, 'title'].tolist()

    return titles_list


def main():
    st.title("📖 Chat with One Paper")

    selected_option = []
    options = []
    text_input = st.text_input("Write a title or topic to start:", value="")
    if "text_input" not in st.session_state or st.session_state.text_input != text_input:
        st.session_state.text_input = text_input
        if text_input:
            d_removed = search_excel(text_input)
            options = remove_duplicates(d_removed)
            st.session_state.options = options
    else:
        if "options" in st.session_state:
          options = st.session_state.options
        else:
          st.warning("Please write a title or topic to start!")

    selected_option = st.selectbox("Select one:", options)

    
    if "selected_option" not in st.session_state or st.session_state.selected_option != selected_option:
      if selected_option:
          st.session_state.selected_option = selected_option
          # Load the document if the selected option changes
          df = pd.read_excel(xlsx_file_path)
          df = df.loc[:, ["author", "uri", "title", "text"]]
          dic = row_to_dict(df, selected_option)
          if dic is not None:
              with open(txt_path, 'w', encoding="utf-8") as txt_file:
                  paper = dic['text']
                  txt_file.write(paper)
              st.session_state.dic = dic
          else:
              st.session_state.dic = {}
              # st.warning("No document found for the selected title. Please select a valid title.")
      else:
          pass
    else:
        dic = st.session_state.dic

    loader = TextLoader(txt_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, separators=[" ", ",", "\n"])
    docs = text_splitter.split_documents(documents)

    # if docs:
    if "selected_option" in st.session_state and "options" in st.session_state:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        if embeddings:
            FAretriever = FAISS.from_documents(docs, embeddings)
            rqa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                              chain_type="stuff",
                                              retriever=FAretriever.as_retriever(search_kwargs={"k": 5}),
                                              verbose=True, )
            text_input2 = st.text_input("""What do you want to know about this paper?
            
Tips: Please be as precise as possible. For example, instead of using 'author', you should say 'authors of this paper'.""",
                                        value="")
            if "text_input2" not in st.session_state or st.session_state.text_input2 != text_input2:
                st.session_state.text_input2 = text_input2
                if text_input2:
                    with st.spinner("Generating answers..."):
                        outp = rqa(text_input2)
                        st.markdown("#### Answer:")
                        st.write(outp["result"])
        else:
            st.warning("Failed to generate embeddings.")
    else:
        st.warning("No valid documents to generate response.")


if __name__ == "__main__":
    main()
