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

# Logic of this code:
#
# 1. User input a title or topic
# 2. Search for the paper using a conversational retrieval chain, to return a list of papers titles and it's metadata,
#    return the top 10 results, remove duplicates, and display the list of titles in a selectbox
# 3. User select a paper title
# 4. Use the selected title to search for the paper in all-paper CSV file, return the paper's author, uri, title
#    and whole text to a list.
# 5. write the text into a text file
# 6. User ask a specific question of the paper content, use the text file as retriever of RetrievalQA chain,
#    to return the answer to the user's question.

# Set up initial configurations
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['pc_api_key']
PINECONE_ENVIRONMENT = os.environ['pc_env']
index_name = os.environ['pc_index']

model_name = 'text-embedding-ada-002'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(index_name)

embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone(index, embed.embed_query, "text")

CONDENSE_PROMPT = """Given the following conversation and a follow up question, print out the follow up question.

Chat History:
{chat_history}
Follow Up Input: {question}
Follow up question:"""

CONDENSEprompt = PromptTemplate(input_variables=["chat_history", "question"], template=CONDENSE_PROMPT)
memory1 = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                          max_token_limit=150,
                                          memory_key='chat_history',
                                          return_messages=True,
                                          output_key='answer')
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                           vectorstore.as_retriever(search_kwargs={"k": 10}),
                                           chain_type="stuff",
                                           memory=memory1,
                                           condense_question_prompt=CONDENSEprompt,
                                           verbose=True,
                                           return_source_documents=True)


class DocumentInput(BaseModel):
    question: str = Field()


def remove_duplicates(input_list):
    return list(set(input_list))


def get_titles_from_dict(result):
    return remove_duplicates([doc.metadata.get('title') for doc in result['source_documents']])


def row_to_dict(df, title):
    row_data = df.loc[df['title'] == title]
    return row_data.to_dict(orient='records')[0] if not row_data.empty else None


def main():
    st.title("📖 Chat with One Paper")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_file_path = os.path.join(current_dir, 'r', 'merged_data02.xlsx')
    txt_path =  os.path.join(current_dir, 'r', '1.txt')

    text_input = st.text_input("Write a title or topic to start:", value="")
    if "text_input" not in st.session_state or st.session_state.text_input != text_input:
        st.session_state.text_input = text_input
        options = []
        if text_input:
            with st.spinner("Searching for papers..."):
                res = qa(text_input)
                options = get_titles_from_dict(res)
            st.session_state.options = options
    else:
        options = st.session_state.options

    selected_option = st.selectbox("Select one:", options)

    if "selected_option" not in st.session_state or st.session_state.selected_option != selected_option:
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
            st.warning("No document found for the selected title. Please select a valid title.")
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
                        # st.markdown(outp["result"])
        else:
            st.warning("Failed to generate embeddings.")
    else:
        st.warning("No valid documents to generate response.")

    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # FAretriever = FAISS.from_documents(docs, embeddings)


if __name__ == "__main__":
    main()
