import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from dotenv import  load_dotenv



with st.sidebar:
    st.title('llm chat with pdf')


def main():
    st.header("chat with pdf")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    # load_dotenv()
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] 
  

 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        # Iterate through the number of pages
        # for page_num in range(len(pdf_reader.pages)):
        #     page = pdf_reader.pages[page_num]
        #     text += page.extract_text() if page.extract_text() else ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200 , 
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        
        storename = pdf.name[:-4]
        index_filepath = f"{storename}_faiss_index"

# Check if the index file already exists
        if not os.path.exists(index_filepath):
            embeddings = OpenAIEmbeddings()

            Vector_store = FAISS.from_texts(chunks , embeddings)

            # If the file does not exist, save the new index
            Vector_store.save_local(index_filepath)
            # st.write("New index created and saved.")
        else:
            # If the file exists, load the existing index
            embeddings = OpenAIEmbeddings()
            Vector_store = FAISS.load_local(index_filepath, embeddings)
            # st.write("Existing index loaded.")

        #Accept user query / question
        query = st.text_input("Ask Question About Your Pdf File")
        if query:
           docs = Vector_store.similarity_search(query=query , k=3)
           llm = ChatOpenAI( model_name = "gpt-4"   )
           chain = load_qa_chain(llm=llm , chain_type="stuff")
           response = chain.run(input_documents= docs  , question= query)
           st.write(response)

        #    st.write(docs)



if __name__ == "__main__":
    main()


