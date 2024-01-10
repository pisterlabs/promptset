import st_pages
import streamlit as st
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain  # type: ignore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

st_pages.add_page_title()  # type: ignore

st.title("Analyze any PDF file with ChatGPT :robot_face:")
openai_key = st.secrets["OPENAI_KEY"]
upload_file = st.file_uploader("Load your own PDF file")

template = """You are an SQLite expert. Given an input question, first create a
syntactically correct SQLite query to run, then look at the results of the query and
return the answer to the input question. Unless the user specifies in the question a
specific number of examples to obtain, query for at most {top_k} results using the LIMIT
clause as per SQLite. You can order the results to return the most informative data in
the database. Never query for all columns from a table. You must query only the columns
that are needed to answer the question. Wrap each column name in double quotes (") to
denote them as delimited identifiers. Pay attention to use only the column names you
can see in the tables below. Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}
"""
if upload_file is not None:
    pdf_reader = PdfReader(upload_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(client=None, openai_api_key=openai_key)

    knowledge_base = FAISS.from_texts(chunks, embeddings)  # type: ignore

    query = st.text_input(
        label="Any questions?", help="Ask any question based on the loaded file"
    )

    if query:
        docs = knowledge_base.similarity_search(query)
        prompt = PromptTemplate(input_variables=["input"], template=template)
        lang_model = OpenAI(
            openai_api_key=openai_key,
            temperature=0,
            max_tokens=300,
            client=None,
        )
        chain = load_qa_chain(lang_model, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)

            st.sidebar.write(  # type: ignore
                f"Your request costs: {str(cb.total_cost)} USD"
            )
        st.write(response)  # type: ignore
