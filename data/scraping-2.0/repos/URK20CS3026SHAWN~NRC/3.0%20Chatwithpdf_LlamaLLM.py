#import the necessary packages
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#Parameters for Metal
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.
# Make sure the model path is correct for your system!
model_path="Models/Vicuna-13B-CoT.Q2_K.gguf" #Path to your Llama/ Llama Derivative Model


def main():
    st.set_page_config(page_title="EduChain", page_icon="📚", layout="wide")
    st.header("Students Helpdesk 💬")
    pdf = st.file_uploader("Upload your Document(PDF)", type="pdf")
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        

      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      embeddings = CohereEmbeddings(cohere_api_key="YOUR-COHERE-API-Key")
      #embeddings = LlamaCppEmbeddings(model_path=model_path)
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        print("Docs: ", docs)
        
        llm = LlamaCpp(
            model_path = model_path,
            max_tokens=1024,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True, # Verbose is required to pass to the callback manager
            n_ctx=2048
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
           
        st.write(response)

if __name__ == "__main__":
    main()

  # Command to Run Script:
  # streamlit run Testwith\ pdf/testwithpdf_crisp.py 