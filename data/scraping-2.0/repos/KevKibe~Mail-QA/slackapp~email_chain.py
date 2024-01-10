import os
import time
from mail_fetch import GmailAPI
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from mail_preprocess import TextProcessor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from supabase import create_client
load_dotenv


class ConversationChain:
    def __init__(self, email):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.email = email
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
    

        
    def fetch_access_token(self):
        """Fetches the access token from the Supabase database."""
        try:
            access_token = self.supabase_client.table('slack_app').select('accesstoken').eq('email', self.email).single().execute()
            return access_token.data['accesstoken']
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def preprocess_emails(self,access_token):
        """Fetching and preprocesses the emails."""
        text_processor = TextProcessor()
        gmail_api = GmailAPI(access_token)
        email_data_list = gmail_api.get_emails(7)
        processed_data = []

        for email_data in email_data_list:
            processed_email_data = text_processor.preprocess_email_data(email_data)
            processed_data.append(str(processed_email_data))
        
        data = ' '.join(processed_data)
        return data
    
        
    def initialize_embeddings_and_vectorstore(self, data):
        """Initializes the embeddings and vectorstore for the chatbot."""
        model_name = 'text-embedding-ada-002'

        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.openai_api_key
        )
        fs = LocalFileStore("./cache/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, fs, namespace=embeddings.model)
                                                                

        chunk_size = 1000
        chunk_overlap = 200
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for dat in data:
            text_chunks = text_splitter.split_text(dat)
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=cached_embedder)
            return vectorstore

    def initialize_conversation_chain(self, vectorstore):
        """Initializes the conversation chain for the chatbot."""
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            model_kwargs={'api_key': self.openai_api_key},
            temperature= 0
        )
        template = """As an AI assistant, I assist with email and workspace data based on provided questions and context. 
                    Company data after a filename and emails are those with tags from, date, subject, labels. 
                    If I can't answer a question, I'll request more information. 
                    Question: {question} {context}
                    Answer:"""
        prompt_template = PromptTemplate(input_variables=["question", "context"], template=template) 
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            # chain_type_kwargs={"prompt": prompt_template},
            memory=memory,
            retriever=vectorstore.as_retriever()
        )
        return conversation_chain

    def run(self, user_input):
        """Runs the chatbot."""
        access_token = self.fetch_access_token()
        data = self.preprocess_emails(access_token)
        vectorstore = self.initialize_embeddings_and_vectorstore(data)
        conversation_chain = self.initialize_conversation_chain(vectorstore)
        return conversation_chain.run(user_input)
        # return data
        
con = ConversationChain("keviinkibe@gmail.com")
prompt = input('>>')
start_time = time.time()
run = con.run(prompt)
end_time = time.time()
print(run)
duration = end_time-start_time
print(duration)