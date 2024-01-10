import pandas as pd
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import streamlit as st
from langchain.document_loaders import UnstructuredExcelLoader

os.environ["OPENAI_API_KEY"] = "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"

def document_QA(df,query):
    loader = UnstructuredExcelLoader(df, mode='elements')
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                        retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    query = query
    response = chain({"question": query})
    return response['result']




def main():

    st.markdown("""<div class="nav">
<h1 style='text-align: center; display: inline-block; color: #fec301;'>QA BASED CHATBOT FOR DOCUMENTS</h1>
</div>""",unsafe_allow_html=True)

#     st.markdown("""<div class="nav" style="display: flex; gap: 10px;">
# <img src="https://romeu.com/wp-content/uploads/2020/10/icono_empresa_ARKAS.png" alt="" height="150px" />
# <h1 style='text-align: left; display: inline-block; color: #fec301;'>QA BASED CHATBOT FOR DOCUMENTS</h1>
# </div>""", unsafe_allow_html=True)

    st.subheader("""Select your document.
                    available documents: 1- 'Profit and Loss.csv'
                         2- 'Service Based Avr SF.csv' """)

    # Soru barını beyaz yapın
    st.markdown(
        """
        <style>
            .stTextInput>div>div>input {
                background-color: #FFFFFF;
                color: #000000;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    user_input = st.text_input("document: ('1' or '2')")

    if user_input:

        if user_input == '1':
            st.write('document 1 is chosen')
            query = st.text_input("Ask question: ")
            if query:
                df="https://raw.githubusercontent.com/mertsengil/mertsengil/main/Profit%20and%20Loss.xlsx"
                response = document_QA(df,query)
                st.write(response)

        elif user_input == '2':
            st.write('document 2 is chosen')
            query = st.text_input("Ask question: ")
            if query:
                df="https://raw.githubusercontent.com/mertsengil/mertsengil/main/Service%20Based%20Avr%20SF.xlsx"
                response = document_QA(df,query)
                st.write(response)
        else:
            st.write("document not found.")

if __name__ == "__main__":
    main()
