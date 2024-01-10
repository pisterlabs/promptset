import os
import pickle
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# Model Names
GPT_35_TURBO = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'

# Prompt
PROMPT_TEMPLATE = """
            Use the following pieces of context to answer the question at the end. 
            You are role playing as Sameer's Personal Assistant. Always reply as a person. NEVER BREAK THE CHARACTER!
            You have your own opinions and knowledge based on the information you have from my resume provided 
            in the context. Never lie. If you dont know the answer then just say that you were not able to find that 
            information in the resume, don't try to make up an answer. You can use the chat history to answer some 
            specific questions if required.
            {context}
            Question: {question}
            chat_history: {chat_history}
        """

QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)


class ResumeChatAPI:
    def __init__(self):
        # Load the ResumePDF for the chatbot
        self.pdf_reader = PdfReader(f"{os.environ.get('RESUME_PATH')}")
        self.chat_history = []
        self.text = ""  # Text from the resume in chunks
        self.text_splitter = None  # Recursive Text Splitter
        self.chunks = None  # Storing Generated Chunks
        self.store_name = "Sameer_Resume"
        self.vector_store = None
        self.llm = None  # Initial LLM is set to None and will be initialized in the preprocess pipeline
        self.retriever = None  # Vector Store Retriever (Will be set in preprocess pipeline)
        self.memory = None  # Memory to keep track of the Chat History
        self.qa = None  # Question and Aswer Chain
        self.__preprocess()  # Preprocess the data, making it ready to accept prompts

    def __preprocess(self):
        # Generate the text out of the model
        for page in self.pdf_reader.pages:
            self.text += page.extract_text()

        # Split the text using RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Generate the chunks
        self.chunks = self.text_splitter.split_text(text=self.text)

        # We don't want to generate the vector multiple times (repetitive work)
        # We use the pickle file if the Vector embeddings are generated already
        if os.path.exists(f"{self.store_name}.pkl"):
            with open(f"{self.store_name}.pkl", "rb") as f:
                self.vector_store = pickle.load(f)
        else:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_texts(self.chunks, embedding=self.embeddings)
            with open(f"{self.store_name}.pkl", "wb") as f:
                pickle.dump(self.vector_store, f)

        # We are ready with the embeddings and the knowledge-base - We are basically ready to accept the questions
        # We want to make sure that these answers are only generated from the text from my resume
        # this can be done by giving it prompt to work with

        self.llm = ChatOpenAI(model_name=GPT_35_TURBO, temperature=0)

        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # create a chain to answer questions
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.retriever,
            verbose=True,
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            memory=self.memory
        )

    def get_answers(self, question, chat_history):
        """
        Get the answers to your question from the PDF File

        :param question: Question asked by the user
        :param chat_history: Maintain history for all the chat messages and reply to be used later
        :return: Answer from the BOT and Chat History
        """
        result = self.qa({"question": question})
        self.chat_history = [*chat_history, (question, result["answer"])]
        return result["answer"], self.chat_history
