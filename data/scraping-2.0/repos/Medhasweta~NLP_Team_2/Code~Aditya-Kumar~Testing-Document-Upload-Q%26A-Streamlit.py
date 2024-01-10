import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import pickle
import os
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
#from unstructured.partition.auto import partition
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering, pipeline, AutoModel
from langchain.chains import RetrievalQA


#with st.sidebar:
    #st.title("Document Upload - Q&A Chat App")

def main():
    st.title("Upload Your Document in PDF format.")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
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

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            #embedding = AutoTokenizer.from_pretrained('google/flan-t5-large')
            #embedding  = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa")
            embedding = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embedding)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)


        query = st.text_input("Ask question about your PDF Document:")
        if query:
            st.write(f"You: ", query)
            docs = VectorStore.similarity_search(query, k=5)
            #llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large",task = "text2text-generation",model_kwargs={'temperature': 0.2, 'max_length':5000}, device=0)
            llm = AutoModel.from_pretrained('lmsys/fastchat-t5-3b-v1.0')
            #llm = HuggingFaceHub(repo_id='lmsys/fastchat-t5-3b-v1.0',model_kwargs={'temperature': 1e-10, 'max_length': 32})
            #memory = ConversationBufferMemory(llm=llm,return_messages=True)
            retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=VectorStore.as_retriever())
            #retriever = VectorStore.as_retriever()
            #chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True))
            #chain = load_qa_chain(llm=llm,chain_type="stuff")
            #response = chain.run(input_documents = docs,question=query)
            while True:
                if query.lower() == "exit":
                    print("Chatbot: Thanks!")
                    break
                #result = chain({"query": query})
                response = retrieval_chain.run(input_documents = docs,query=query)
                #response = result["result"]

            st.write("Chatbot:", response)



if __name__ == '__main__':
    main()