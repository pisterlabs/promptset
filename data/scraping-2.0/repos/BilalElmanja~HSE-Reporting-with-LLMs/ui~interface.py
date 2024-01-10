import os
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
import PyPDF2
from io import BytesIO
import re

import replicate
import pdfplumber
import time
from datetime import datetime

class LLM_model(LLM):
    """Together large language models."""

    model: str = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
    """model endpoint to use"""
    together_api_key: str = "r8_4bwvRHrNTm3qpgVHNjO3UdEeY69b8CT0tH7W4"
    """Together API key"""
    temperature: float = 0.01
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "replicate"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""

        output = replicate.run(
            self.model,
            input = {
                    "debug": False,
                    "top_k": 10,
                    "top_p": 1,
                    "temperature": 0.01,
                    "prompt": prompt,
                    "system_prompt": "",
                    "max_new_tokens": 10000,
                    "min_new_tokens": -1
            }
            
        )
        response = ""
        for text in output:
            response += text
        return response
    
    def predict(self, prompt):
        os.environ["REPLICATE_API_TOKEN"] = "r8_4bwvRHrNTm3qpgVHNjO3UdEeY69b8CT0tH7W4"
        api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 1,
                "prompt": "Can you write a poem about open source machine learning? Let's make it in the style of E. E. Cummings.",
                "temperature": 0.5,
                "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                "max_new_tokens": 500,
                "min_new_tokens": -1
            }
            )
        print("\n \n \n")
        print(output)
        return output

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llm"

        
class DocumentChain:
    def __init__(self):
        self.docs = None
        self.model = None
        self.chain = None
        self.retriever = None
        self.bge_embeddings = None
        self.vectorstore = None
        self.store = None
        self.done = False
        self.doc_names = []
        self.directory = "./documents"  

    def init_embedding_model(self):
        self.bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} # set True to compute cosine similarity
        )


    def load_documents(self):
        loaders = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if filename.endswith('.txt'):
                loaders.append(TextLoader(filepath))
            elif filename.endswith('.pdf'):
                loaders.append(PyPDFLoader(filepath))
            self.doc_names.append(filename)
        self.docs = []
        for loader in loaders:
            self.docs.extend(loader.load())

        print(f"Loaded {len(self.docs)} documents")


    def init_retriever(self):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        self.vectorstore = Chroma(collection_name="split_parents", embedding_function=self.bge_embeddings, persist_directory="doc_db/")
        #self.store = InMemoryStore()
        fs = LocalFileStore("./doc_store")
        self.store = create_kv_docstore(fs)
        self.retriever = ParentDocumentRetriever(
            vectorstore= self.vectorstore,
            docstore= self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,

        )
        
        if self.docs:
            self.retriever.add_documents(self.docs)
            
    
    
    def init_model(self):
        self.model = LLM_model()

        
    def init_chain(self):
        self.done = False
        self.init_embedding_model()
        self.init_model()
        if len(os.listdir("./documents")) != 0:
            self.load_documents()
        self.init_retriever()
        #self.chain = RetrievalQA.from_chain_type(llm=self.model,
        #                         chain_type="stuff",
        #                         retriever=self.retriever)
        self.done = True

    
    def add_new_document(self, filename):
        self.done = False
        # check for every doc in directory, if it's already in the store, if not, add it
        filepath = os.path.join(self.directory, filename)

        loader = []
        loader.append(TextLoader(filepath))
        doc = []
        for load in loader:
            doc.extend(load.load())    
        self.retriever.add_documents(doc)
        
        self.done = True
        
    


class ConversationChain():
    def __init__(self):
        super().__init__()
        self.current_conversation_id = None
        self.current_conversation_file = None
        self.directory = "./conversations"
        self.docs = None
        self.model = None
        self.chain = None
        self.retriever = None
        self.bge_embeddings = None
        self.vectorstore = None
        self.store = None
        self.done = False
        self.conv_names = []
        self.embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.retriever2= None
        self.memory_vectorstore = None
        self.store2 = None
        self.memory = None
        

    def start_new_conversation(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        conversation_filename = f"conversation_{self.current_conversation_id}.txt"
        self.current_conversation_file = os.path.join(self.directory, conversation_filename)
        with open(self.current_conversation_file, 'w') as file:
            file.write(f"Conversation ID: {self.current_conversation_id}\n")

    def add_to_conversation(self, conversation):
        with open(self.current_conversation_file, 'a') as file:
            file.write(f"{conversation}")

    def init_embedding_model(self):
        self.bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} # set True to compute cosine similarity
        )


    def load_documents(self):
        loaders = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if filename.endswith('.txt'):
                loaders.append(TextLoader(filepath))
        self.docs = []
        for loader in loaders:
            self.docs.extend(loader.load())

        print(f"Loaded {len(self.docs)} conversations")


    

    def init_retriever(self):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        memory_parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        memory_child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        fs = LocalFileStore("./conv_store")
        self.store = create_kv_docstore(fs)
        self.vectorstore = Chroma(collection_name="old_conv", embedding_function=self.bge_embeddings, persist_directory="conv_db/")
        self.memory_vectorstore = Chroma(collection_name="current_conv", embedding_function=self.bge_embeddings)
        
        self.retriever = ParentDocumentRetriever(
            vectorstore= self.vectorstore,
            docstore= self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,

        )
        self.memory = ParentDocumentRetriever(
            vectorstore= self.memory_vectorstore,
            docstore= InMemoryStore(),
            child_splitter=memory_parent_splitter,
            parent_splitter=memory_child_splitter,

        )
        if self.docs:
            self.retriever.add_documents(self.docs)


    def init_model(self):
        self.model = LLM_model()

        
    def init_chain(self):
        self.done = False
        self.init_embedding_model()
        self.init_model()
        if len(os.listdir("./conversations")) != 0:
            self.load_documents()
        self.init_retriever()
        #self.chain = RetrievalQA.from_chain_type(llm=self.model,
        #                         chain_type="stuff",
        #                         retriever=self.retriever)
        self.done = True

    
    def add_new_document(self, filename):
        self.done = False
        # check for every doc in directory, if it's already in the store, if not, add it
        filepath = os.path.join(self.directory, filename)
        if filename.endswith('.txt'):
            if filename not in self.conv_names:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                doc = Document(
                    name=filename,
                    content=content,
                    metadata={"source": "text", "name" : filename, "path" : filepath}
                )
                self.retriever.add_documents([doc])
                self.conv_names.append(filename)
        
        self.done = True

    def add_to_memory(self):
        # add conversation file to memory
        loader = []
        loader.append(TextLoader(self.current_conversation_file))
        doc = []
        for load in loader:
            doc.extend(load.load())    
        self.memory.add_documents(doc)




class PromptTemplate:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # additional initialization

    def generate_response(self, user_input, retrieved_docs, model):
        
        return self.create_response(user_input, retrieved_docs, model)
        

    def is_relevant(self, user_input , docs, model):
        # Use LLM to generate a response based on the query and docs
        model_prompt = f"below is a provided content extracted from some document in the database, \n " + \
         f"content : \n {docs} \n \n now based on this content, is it sufficient to answer to this question even to get just a little close : {user_input} ? \n answer only by yes or no"
        response = model.predict(model_prompt)
        if "yes" in response:
            return True
        else:
            return False # placeholder

    def create_response(self, user_input, docs, conversation, current_conversation, model):
        # Use LLM to generate a response based on the query and docs
        model_prompt = f"below is a provided content extracted from some document in the database, and it is the most relevant content to my question, \n" + \
         f"content : \n {docs} \n and here's the conversations history between you and me \n conversation : {conversation} \n, and this is our current conversation : \n {current_conversation} \n now based on this content, and the conversations, can you please give an answer to the question : {user_input} " + \
             f" \n  if you find the content not sufficient to answer the question, answer based on the conversation only and in a friendly way, give only the response without any additional details"
        
        return model.predict(model_prompt)  # placeholder
        

    def ask_for_clarification(self):
        return "Can you please provide more specific details?"



class DocumentPromptTemplate:

    def generate_reformulations(self, user_input):
        # Ask the model to provide 4 reformulations of the user's request
        reformulations = self.get_reformulations(user_input, 3)

        # Retrieve relevant documents for each reformulation
        all_reformulations = []
        reformulations = reformulations.split("\n")
        for reformulation in reformulations:
            if "?" in reformulation:
                all_reformulations.append(reformulation)
        
        return all_reformulations

    def get_reformulations(self, input_text, num_reformulations):
        # Implement logic to get reformulations from the model
        return f"here's a question provided by me : {input_text} \n can you please provide {num_reformulations} reformulations of this question ? give each reformulation in a new line" 

    def fuse_documents(self, contexts, user_input):
        # Implement logic to fuse documents into a single context
        return f"here's a provided context from many pieces of documents : \n context 1 : {contexts[0]} \n context 2 : {contexts[1]}  \n now based on this context, can you please provide an answer to the question : {user_input} ?"
        

class ConversationPromptTemplate:

    def generate_reformulations(self, user_input):
        # Ask the model to provide 4 reformulations of the user's request
        reformulations = self.get_reformulations(user_input, 1)

        # Retrieve relevant documents for each reformulation
        all_reformulations = []
        reformulations = reformulations.split("\n")
        for reformulation in reformulations:
            if "?" in reformulation:
                all_reformulations.append(reformulation)
        
        return all_reformulations

    def get_reformulations(self, input_text, num_reformulations):
        # Implement logic to get reformulations from the model
        return f"here's a question provided by me : {input_text} \n can you please provide {num_reformulations} reformulations of this question that represent all my potential meanings ? give each reformulation in a new line" 

    def fuse_documents(self, contexts, user_input):
        # Implement logic to fuse documents into a single context
        return f"here's a provided context from many parts of your conversation with me : \n context 1 : {contexts[0]} \n context 2 : {contexts[1]}  \n now based on this context, can you please provide an answer to the question : {user_input} ? \n if you find the context not sufficient to answer the question, answer only based on my question and forget the conversation context"
        

def convert_pdf_to_text(pdf_file_path, text_file_path=None):
    # Determine the output text file path
    directory, pdf_filename = os.path.split(pdf_file_path)
    base_filename = os.path.splitext(pdf_filename)[0]
    text_file_path = os.path.join("./documents", f"{base_filename}.txt")

    # Open the PDF and extract text
    with pdfplumber.open(pdf_file_path) as pdf, open(text_file_path, 'w', encoding='utf-8') as text_file:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_file.write(text + "\n")

    return text_file_path


def clean_text_file(file_path):
    # Define a regular expression pattern for allowed characters (English and French)
    pattern = re.compile(r"[a-zA-Z0-9\séàèùâêîôûçëïüÉÀÈÙÂÊÎÔÛÇËÏÜ.,!?'\-]")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Filter out unwanted characters
    filtered_content = ''.join(pattern.findall(content))

    # Write the cleaned content back to the file or to a new file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(filtered_content)

    return file_path



import chainlit as cl

# -----------------------------------------------------------------------------------------------

chain = DocumentChain()
conversation_chain = ConversationChain()
prompt_template = PromptTemplate()
document_prompt_template = DocumentPromptTemplate()
conversation_prompt_template = ConversationPromptTemplate()

# -----------------------------------------------------------------------------------------------

# check if the conversation file exists
if not os.path.exists("./conversations"):
    os.mkdir("./conversations")

if not os.path.exists("./documents"):
    os.mkdir("./documents")

# -----------------------------------------------------------------------------------------------

conversation_chain.start_new_conversation()
conversation_chain.init_chain()

# -----------------------------------------------------------------------------------------------

chain.init_chain()



# -----------------------------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    # Example usage
    #msg = cl.Message(content="Hello Dear !")
    #await msg.send()
    #time.sleep(1)
    #files = None 
    # Wait for the user to upload a PDF file
    #if len(os.listdir("./documents")) == 0:

     #   while files is None:
     #       files = await cl.AskFileMessage(
      #          content="Please upload your first PDF File to begin!",
       #         accept=["application/pdf"],
       #         max_size_mb=20,
        #        timeout=500,
        #    ).send()

        #for file in files:
            # Convert the PDF file to text
         #   msg = cl.Message(content=f"Processing `{file.name}`...")
         #   await msg.send()

            # Read the PDF file
        #    pdf_stream = BytesIO(file.content)
        #    pdf = PyPDF2.PdfReader(pdf_stream)
        #    pdf_text = ""
        #    for page in pdf.pages:

        #        pdf_text += page.extract_text()
            # Write the PDF text to a text file
         #   txt_filename = f"{file.name.split('.')[0]}.txt"
         #   txt_filepath = os.path.join("./documents", txt_filename)
         #   with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
         #       txt_file.write(pdf_text)
            
         #   # Clean the text file
         #   clean_text_file(txt_filepath)

    # Let the user know that the model is being initialized
    #while not chain.done:
    #    msg.content = f"Initializing Application ..."
    #    await msg.update()
    #    time.sleep(1)  
        
    #while not conversation_chain.done:
    #    msg.content = f"Initializing Application ..."
    #    await msg.update()
     #   time.sleep(1) 

    # Let the user know that the system is ready
    #msg.content = f"Done ! Ask me anything!"
    #await msg.update()

    cl.user_session.set("chain", chain)
    cl.user_session.set("conversation_chain", conversation_chain)

@cl.on_message
async def on_message( message: cl.Message):
    
    current_conversation = ""
    user_input = message.content
    chain = cl.user_session.get("chain")
    conversation_chain = cl.user_session.get("conversation_chain")
    msg = cl.Message(content="processing ... ")
    await msg.send()

    # check if the user has uploaded a file
    files = message.elements
    if len(files) > 0:
        for file in files:
            print(file)
            # Convert the PDF file to text
            msg.content = f"Processing `{files[0].name}`..."
            await msg.update()

            # Read the PDF file
            pdf_stream = BytesIO(files[0].content)
            pdf = PyPDF2.PdfReader(pdf_stream)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()

            # Write the PDF text to a text file
            txt_filename = f"{file.name.split('.')[0]}.txt"
            txt_filepath = os.path.join("./documents", txt_filename)
            with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write(pdf_text)
            
            # Clean the text file
            clean_text_file(txt_filepath)
            chain.done = False
            await cl.make_async(chain.add_new_document)(file.name.split('.')[0] + ".txt")

    if current_conversation == "":
        #conversation_chain.memory.save_context({"input": user_input}, {"output": ""})
        current_memory = ""
    else:
        try:
            current_memory = conversation_chain.memory.get_relevant_documents(user_input)[0].page_content
            print("\n \n")
            print(" current memory " , current_memory)
            print("\n \n")
        except:
            current_memory = ""

    current_conversation += f"Me: {user_input}\n"

    if chain.done and conversation_chain.done:
        response_doc =  chain.retriever.get_relevant_documents(user_input)
        response_conv =  conversation_chain.retriever.get_relevant_documents(user_input)
        model_response = await cl.make_async(prompt_template.create_response)(user_input, response_doc[0].page_content, response_conv[0].page_content, current_memory, chain.model)
        msg.content = "\n \n"
        await msg.update()
        response = ""

        for text in model_response:
            await msg.stream_token(text)
            response += text
            time.sleep(0.1)

        await msg.send()
        current_conversation += f" \nYou: {response}\n \n"
        conversation_chain.add_to_conversation(current_conversation)
        await cl.make_async(conversation_chain.add_to_memory)()
    

    
