import os

# render UI
import gradio

from typing import Union

# convert doc to pdf
from docx2pdf import convert
# to create qa chain
from langchain.chains import RetrievalQA
# read document
from langchain.document_loaders import (
    # for CSV files
    CSVLoader, 
    # for PDF files
    PyPDFLoader
)
# doc and query embeddings using SentencTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# llm model
from langchain.llms import CTransformers
# chunk large documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# vector store to save document and embeddings
from langchain.vectorstores import FAISS


class ChatApp:
    """
    Llama2 as LLM implemented using Langchain and CTransformers
    """
    def __init__(self):
        self.llama_model = "llama-2-7b-chat.ggmlv3.q8_0.bin"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    def load_document(self, file_name: str) -> FAISS:
        """
        read the document and save to vector datastore

        Args:
            file_name (str): path to file

        Returns:
            vectorstore (FAISS): vector datastore
        """

        # check file extension
        file_extension = file_name.split(".")[-1]
        # read the file using Langchain
        data = ''
        if file_extension == 'pdf' or file_extension == 'docx':
            # convert docx to pdf
            if file_extension == 'docx':
                base_file_name = os.path.basename(file_name)
                # PDF file name
                pdf_file_path = f'/tmp/{base_file_name}.pdf'
                # use docx to convert to PDF
                convert(file_name, pdf_file_path)
                # update file name
                file_name = pdf_file_path
            data = PyPDFLoader(file_name).load()
        elif file_extension == 'csv':
            data = CSVLoader(file_path=file_name, encoding='utf-8').load()
        else:
            print(f"Cannot process {file_extension}. Only 'pdf', 'docx' and 'csv' are supported")
            return None
        
        # perform document chunking        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500, 
            chunk_overlap = 50
        )
        text_chunks = text_splitter.split_documents(data)
        try:
            # get list of embeddings for document
            embeddings = HuggingFaceEmbeddings(
                model_name = self.embedding_model, 
                # run the model on CPU
                model_kwargs = {'device':'cpu'}
            )
            # save document and its embeddings to vector store
            vectorstore = FAISS.from_documents(
                documents = text_chunks, 
                embedding = embeddings
            )
            return vectorstore
        except Exception as error:
            print(f"error while creating vector store :: Exception:: {str(error)}")
            return None

    def get_answer(self, fileobj: gradio.File, search_query: str) -> Union[str, list]:
        """
        get answers to user question from downloaded Llama2 weights

        Args:
            fileobj : Gradio file object of uploaded file
            search_query (str): user question
        
        Returns:
            answer (str): answer from the LLM
            sources (list): source to answer
        """

        answer, sources = 'could not generate an answer', []
        
        # get the vector datastore
        db = self.load_document(file_name=fileobj.name)
        if db:
            try:
                llm = CTransformers(
                    model = self.llama_model, 
                    model_type = 'llama', 
                    max_new_tokens = 512, 
                    temperature = 0
                )
                chain = RetrievalQA.from_chain_type(
                    llm = llm, 
                    # TODO- can try an alternate chain type
                    chain_type = "stuff", 
                    # get only the most relevant document
                    retriever = db.as_retriever(search_kwargs={'k': 1}),
                    return_source_documents = True
                )
                print("Fetching results ...")
                result = chain({"query": search_query})
                answer = result.get('result')
                print('answer: ', answer)
                sources = result.get('source_documents')
            except Exception as error:
                print(f"error rendering answer :: Exception :: {str(error)}")
        
        return answer, sources

def gradio_interface(inputs: list=[gradio.File(label = "Input file", file_types = [".pdf", ".csv", ".docx"]), 
                                   gradio.Textbox(label = "your input", lines = 3, placeholder = "Your search query ...")], 
                     outputs: list=[gradio.Textbox(label = "response", lines = 6, placeholder = "response returned from llama2 ..."), 
                                    gradio.Textbox(label = "response source", lines = 6, placeholder = "source document from vector store ...")]):
    """
    render a gradio interface

    Args:
        inputs (list): interface input components
        outputs (str): output generated by llama2
    """

    chat_ob = ChatApp()
    demo = gradio.Interface(fn = chat_ob.get_answer, inputs = inputs, outputs = outputs)
    demo.launch(share = False)
    # uncomment for public URL (accessible for 3 days, deploy and it is accessible for a lifetime)
    # demo.launch(share = True)


if __name__ == '__main__':

    gradio_interface()
