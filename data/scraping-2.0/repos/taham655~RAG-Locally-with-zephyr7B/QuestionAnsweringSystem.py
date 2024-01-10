# Importing necessary libraries and modules
import os
import gradio as gr  # Gradio for creating web interfaces
from langchain.callbacks.manager import CallbackManager  # For managing callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # For streaming stdout callbacks
from langchain.chains import LLMChain, RetrievalQA, ConversationChain  # Various chain types from langchain
from langchain.llms import LlamaCpp  # Llama language model implementation
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain.vectorstores import Chroma  # For vector storage and retrieval
from langchain.memory import ConversationBufferWindowMemory  # For conversation memory handling
from langchain.embeddings import HuggingFaceBgeEmbeddings



class QuestionAnsweringSystem:
    def __init__(self, model_path, store_directory="stores/lit_cosine"):
        # Constructor for the QuestionAnsweringSystem class

        # Initializing a callback manager with a streaming stdout callback handler
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Initializing the Llama language model with specified parameters
        self.llm = LlamaCpp(
            model_path=model_path,  # Path to the model
            n_gpu_layers=1,  # Number of GPU layers to use
            n_batch=512,  # Batch size for processing
            f16_kv=True,  # Use 16-bit floating point for key/value pairs for efficiency
            callback_manager=self.callback_manager  # Assigning the callback manager
        )
        print("LLM Initialized...")  # Confirmation message for successful initialization

        # Defining the prompt template for the QA system
        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else. If the answer of the question is not in the context, just say "I don't know". Please do not make up an answer or try to give it using your own knowledge.
        Helpful answer:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  # Creating a prompt template object

        # Initializing the DocumentProcessor for embedding functionality
        embeddings = HuggingFaceBgeEmbeddings(
            model_name = "BAAI/bge-large-en",
            model_kwargs = {'device':'cpu'},
            encode_kwargs = {'normalize_embeddings':False}
        )

        self.load_vector_store = Chroma(persist_directory=store_directory, embedding_function=embeddings)

        self.retriever = self.load_vector_store.as_retriever(search_kwargs={"k": 1})  # Setting up the retriever


        # Initializing the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt},
            verbose=False
        )

    def answer_query(self, query):
        # Method to process a query and return the answer
        if query.lower() == "exit":  # Exit condition
            return "Exiting..."
        
        print(self.retriever.get_relevant_documents(query))
        response = self.qa(query)  # Process the query using the QA chain
        return response['result']  # Return the result

