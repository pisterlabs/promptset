import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import os

# streamlit web\main2.py
# pip install faiss-cpu or faiss-gpu

with st.sidebar:
    st.title('💬 Chat App')
    st.markdown('''
        ## About
        Single document chatbot
        ''')
    add_vertical_space(3)
load_dotenv()
history = []


def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF Document", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        if not chunks:
            st.markdown(f"""
            :red[Unable to extract text from `{pdf.name}`]

            Please try again with a different PDF document.
            """)
            st.stop()
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        # Accept user questions/query
        query = st.text_input("Ask questions about your Document:")
        current_history = history[::-1]
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                history.append(f"""
                                    :blue[{query}]"


                                    {response}



                                    """)
                print(cb)
            st.write(response)
            if current_history:
                st.markdown("""---""")
                st.markdown("".join(current_history))


if __name__ == "__main__":
    main()
