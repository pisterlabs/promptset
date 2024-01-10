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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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
from openai import OpenAI

client = OpenAI(
            api_key="API_KEY",
 )


def create_assistant():
    
	
    my_assistant = client.beta.assistants.create(name = "Jone Albert",
                        instructions = "you are an HSE expert," ,
                        model="gpt-4-1106-preview",
                        tools=[{"type": "retrieval"}],

    )
    return my_assistant

my_assistant = create_assistant()

def initiate_interaction(user_message):
    
	my_thread = client.beta.threads.create()
	message = client.beta.threads.messages.create(thread_id=my_thread.id,
                                              	role="user",
                                              	content=user_message,
                                              	
	)
    
	return my_thread


my_thread = initiate_interaction(" your name is Jone Albert, and your job is to interrogate me with a bunch of questions to know my working situation relative to HSE and to generate a final report, the questions should be from general to specific, Now give the first question. start by greeting me, and then ask the question.")

run = client.beta.threads.runs.create(
thread_id = my_thread.id,
assistant_id = my_assistant.id,
        )

while run.status != "completed":
    
    
    keep_retrieving_run = client.beta.threads.runs.retrieve(
    thread_id=my_thread.id,
    run_id=run.id
    )
    print(f"Run status: {keep_retrieving_run.status}")

    if keep_retrieving_run.status == "completed":

        print("\n")
        break








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
        #os.environ["REPLICATE_API_TOKEN"] = "r8_4bwvRHrNTm3qpgVHNjO3UdEeY69b8CT0tH7W4"

        my_thread = initiate_interaction(prompt)

        run = client.beta.threads.runs.create(
  	    thread_id = my_thread.id,
  	    assistant_id = my_assistant.id,
	            )

        while run.status != "completed":
         
         
            keep_retrieving_run = client.beta.threads.runs.retrieve(
            thread_id=my_thread.id,
            run_id=run.id
            )
            print(f"Run status: {keep_retrieving_run.status}")

            if keep_retrieving_run.status == "completed":

                print("\n")
                break

        messages = client.beta.threads.messages.list(thread_id=my_thread.id)

        response = messages.data[0].content[0].text.value
        return response
        #api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
        #output = replicate.run(
        #"meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
        #input={
        #    "debug": False,
        #    "top_k": 5,
        #    "top_p": 1,
        #    "prompt": prompt,
        #    "temperature": 0.01,
        #    "max_new_tokens": 2048,
        #    "min_new_tokens": -1
        #}
        #)
        #for text in output:
        #    print(text)
    #
        
        
        #response = client.completions.create(
        #        model="gpt-3.5-turbo-instruct",
        #        prompt=prompt,
        #        max_tokens=512,
        #        temperature=0
        #        )
        

        #return response.choices[0].text


        

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

    def update_processed_docs(self):
        with open("processed_docs.txt", 'w') as file:
            for doc_name in self.doc_names:
                file.write(doc_name + "\n")

    def load_documents(self):
        loaders = []
        for filename in os.listdir(self.directory):
            if filename in self.doc_names:
                continue
            else:
                filepath = os.path.join(self.directory, filename)
                if filename.endswith('.txt'):
                    loaders.append(TextLoader(filepath))
                elif filename.endswith('.pdf'):
                    loaders.append(PyPDFLoader(filepath))
                self.doc_names.append(filename)
        self.docs = []
        for loader in loaders:
            self.docs.extend(loader.load())

        # write a text file in the current directory with the names of the documents already processed
        self.update_processed_docs()


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
        self.doc_names.append(filename)
        self.update_processed_docs
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


class Question_State_Chain:
    # this is a class that controls the flow of a question answering bot, the bot will question the user, and based on the user's response, it will decide what to do next
    def __init__(self, model):
        self.state = "start"
        self.question = ""
        self.response = ""
        self.question_answered = False
        self.conversation = ""
        self.count_questions = 0
        self.model = model
        
    
    def start(self):
        self.question = "you are an HSE expert, your name is Jone Albert, and your job is to interrogate me with a bunch of questions to know my working situation relative to HSE and to generate a final report, the questions should be from general to specific, " + \
        "Now give the first question. start by greeting me, and then ask the question."
        response = self.model.predict(self.question)
        return response
        
 
    def complete_conversation(self):
        
        prompt = f"you are Jone Albert, based on the conversation history between you and me : \n{self.conversation}, \n \nif we didn't complete 4 questions, ask the next relevant question, don't talk about the previous questions, just ask the next relevant question. " + \
            f"if you already asked 4 questions and i answered all of them (no matter what the answer, you only want to know better my working situation the company), write a full report within the HSE regulations." 
        response = self.model.predict(prompt)
        return response
    
    def generate_report(self):
        prompt = f"you are Jone Albert, based on the conversation history between you and me : \n{self.conversation}, \n \nwrite a full report within the HSE regulations" 
        response = self.model.predict(prompt)
        return response
    


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


# Function to create a PDF file
def create_pdf(file_path, text):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    margin = 35  # 1 inch margin

    text_object = c.beginText(margin, height - margin)
    text_object.setFont("Times-Roman", 12)
    text_object.textLines(text)

    c.drawText(text_object)
    c.save()

# Path where the PDF will be saved
pdf_file_path = "final_report.pdf"


import chainlit as cl

# -----------------------------------------------------------------------------------------------

document_chain = DocumentChain()
conversation_chain = ConversationChain()
prompt_template = PromptTemplate()


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

document_chain.init_chain()
question_state_chain = Question_State_Chain(model=document_chain.model)

# -----------------------------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():

    cl.user_session.set("document_chain", document_chain)
    cl.user_session.set("conversation_chain", conversation_chain)
    cl.user_session.set("question_state_chain", question_state_chain)

    msg = cl.Message(content="Welcome to HSE Report Agent !\n Type 'okey' to start !")
    await msg.send()
    
    
@cl.on_message
async def on_message( message: cl.Message):

    question_state_chain = cl.user_session.get("question_state_chain")
    document_chain = cl.user_session.get("document_chain")
    conversation_chain = cl.user_session.get("conversation_chain")
    msg = cl.Message(content="processing ... ")
    await msg.send()
    # user type anything to start the conversation
    if question_state_chain.state == "start":
        msg.content += "\n \n"
        await msg.update()
        question = await cl.make_async(question_state_chain.start)()
        quesion_text = ""
        for text in question:
            await msg.stream_token(text)
            quesion_text += text
            time.sleep(0.01)

        await msg.send()
        question_state_chain.conversation += f"You: {quesion_text}\n \n"
        question_state_chain.question = quesion_text
        question_state_chain.state = "ask_question"

    else:
        
        user_input = message.content
        if user_input == "report":
            question_state_chain.state = "report"
            response = await cl.make_async(question_state_chain.generate_report)()
            response_text = ""
            msg.content = ""
            await msg.update()
            for text in response:
                response_text += text
                await msg.stream_token(text)
                time.sleep(0.01)

            
            # save the report in a pdf file
            with open(f"report_{conversation_chain.current_conversation_id}.pdf", 'w', encoding='utf-8') as file:
                file.write(response_text)


            # Create the PDF
            create_pdf(pdf_file_path, response_text)
            # send the pdf file to the user
            msg.content += "\n\nHere's the downloadable version of the report"  
            await msg.update()
            elements = [
                            cl.File(
                        name="final_report.pdf",
                        path="./final_report.pdf",
                        display="inline",
                        ),
                ]  
            msg.elements = elements



            await msg.send()
            
            #question_state_chain.conversation += f"You: {response_text}\n \n"
            #question_state_chain.state = "end"
            return
        question_state_chain.conversation += f"Me: {user_input}\n \n"

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
                document_chain.done = False
                document_chain.add_new_document(file.name.split('.')[0] + ".txt")

        
        try:
            current_memory = conversation_chain.memory.get_relevant_documents(user_input)[0].page_content
            print("\n \n")
            print(" current memory " , current_memory)
            print("\n \n")
        except:
            current_memory = ""

        

        response_value = await cl.make_async(question_state_chain.complete_conversation)()
        response_value_text = ""
        msg.content = ""
        await msg.update()
        for text in response_value:
            response_value_text += text
            await msg.stream_token(text)
            time.sleep(0.01)

        question_state_chain.conversation += f"You: {response_value_text}\n \n"
        

       # print("\n passed to get_QA_value_response \n")
       # question = await cl.make_async(question_state_chain.get_QA_value_response)(question_state_chain.conversation)
       # question_text = ""
        #msg.content += "\n\n"
        #await msg.update()
        #for text in question:
        #    question_text += text
        #    await msg.stream_token(text)
        #await msg.send()
        #question_state_chain.question = question_text
        #question_state_chain.conversation += f"You: {question_text}\n \n"
        

        #response_doc =  document_chain.retriever.get_relevant_documents(user_input)
        #response_conv =  conversation_chain.retriever.get_relevant_documents(user_input)
        #model_response = await cl.make_async(prompt_template.create_response)(user_input, response_doc[0].page_content, response_conv[0].page_content, current_memory, document_chain.model)
        #msg.content = "\n \n"
        #await msg.update()
        #response = ""

        #for text in model_response:
        #    await msg.stream_token(text)
        #    response += text
        #    time.sleep(0.1)

        await msg.send()
        #current_conversation += f" \nYou: {response}\n \n"
        #conversation_chain.add_to_conversation(current_conversation)
        #conversation_chain.add_to_memory()
        # write the conversation to a text file in the conversations directory
        conversation_chain.start_new_conversation()
        with open(conversation_chain.current_conversation_file, 'a') as file:
            file.write(f"{question_state_chain.conversation}")


    
