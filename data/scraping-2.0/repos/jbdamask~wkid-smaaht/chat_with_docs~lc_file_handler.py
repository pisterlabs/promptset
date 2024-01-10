import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, load_tools
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredFileLoader, WebBaseLoader
# from custom_agent_types import CustomAgentType
import pandas as pd
import abc
import os
import requests
from io import StringIO
from src.logger_config import get_logger
from langchain.vectorstores import Chroma
from langchain.vectorstores import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# Necessary for ChromaDB to work on Fargate linux instances
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Necessary for ChromaDB to work on Fargate linux instances
langchain.verbose = True
# Configure logging
logger = get_logger(__name__)

class Handler(abc.ABC):

    first_impression = "What's this file about?"

    def __init__(self, file, openai_api_key, slack_bot_token):
        # _, file_extension = os.path.splitext(file)
        if not isinstance(file, str):
            _, file_extension = os.path.splitext(file.get('name'))
            self.file_type = file_extension.lstrip('.').lower()
        self.file = file
        self.openai_api_key = openai_api_key
        # self.slack_bot_token = slack_bot_token
        self.headers = {'Authorization': f'Bearer {slack_bot_token}'}
        self.llm = ChatOpenAI(temperature=0,openai_api_key=self.openai_api_key)

    # def handle(self, file):
    def handle(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def instantiate_loader(self, filename):
        pass

    # TODO - I think I can remove this
    def _read_file_content(self, url, SLACK_BOT_TOKEN) :
        headers = {'Authorization': f'Bearer {SLACK_BOT_TOKEN}'}
        response = requests.get(url, headers=headers)
        return response.content  

    def download_and_store(self):
        # headers = {'Authorization': f'Bearer {self.slack_bot_token}'}
        url = self.file.get('url_private')
        logger.info(url)
        filepath = self.download_local_file()        
        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.instantiate_loader(filepath)
        documents = self.loader.load()
        self.docs = text_splitter.split_documents(documents)
        filename = self.file.get('name')
        for idx, text in enumerate(self.docs):
            self.docs[idx].metadata['filename'] = filename.split('/')[-1]   
        filtered_docs = utils.filter_complex_metadata(self.docs)
        # self.db = Chroma.from_documents(docs, embeddings)    
        self.db = Chroma.from_documents(filtered_docs, embeddings)    
        self.delete_local_file(filepath)    


    # TODO - May or may not want to keep this method
    def q_and_a(self, question):
        # Assumes an agent has been configured. 
        result = None
        try:
            result = self.agent.run(question)
        except Exception as e:
            logger.error(e)
            result = "Sorry, I ran into a problem with the file"
        return result
        # return self.agent(question) # This invokes the default __call__ method

    # Not all filetypes are accessible by LangChain over the web.
    # Some need to be downloaded locally
    # def _download_local_file(self, headers, directory='downloads'):
    def download_local_file(self):    
        import requests
        import uuid
        directory='downloads'
        url = self.file.get('url_private')
        file_type = url.split('.')[-1]
        # response = requests.get(url, headers=headers)
        response = requests.get(url, headers=self.headers)
        # Generate a random UUID
        file_uuid = uuid.uuid4()
        # Convert the UUID to a string and append the .docx extension
        filename = str(file_uuid) + '.' + file_type
        # Check if the directory exists and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return filepath

    # You slob. Clean up after yourself!
    def delete_local_file(self, filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)

class PDFHandler(Handler):

    def handle(self):
        return f"Handling PDF file: {self.file}"

    def instantiate_loader(self, filename):
        # self.loader = UnstructuredPDFLoader(filename, mode="elements", metadata_filename=self.file.get('url_private'))
        # self.loader = PyPDFLoader(filename, metadata_filename=self.file.get('url_private'))
        self.loader = PyPDFLoader(filename)

class DOCXHandler(Handler):
    def handle(self):
        return f"Handling DOCX file: {self.file}"

    def instantiate_loader(self, filename):
        self.loader = UnstructuredWordDocumentLoader(filename, mode="elements")         

class TxtHandler(Handler):
    def handle(self):
        return f"Handling txt file: {self.file}"  
    
    def instantiate_loader(self, filename):
        self.loader = UnstructuredFileLoader(filename, mode="elements")
      

class WebHandler(Handler):

    def handle(self):
        return f"Handling web page: {self.file}"
    
    def instantiate_loader(self, filename):
        self.loader = WebBaseLoader(filename)
    
    def load_split_store(self):
        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # self.instantiate_loader(self.file.get('url_private'))
        self.instantiate_loader(self.file)
        documents = self.loader.load()
        self.docs = text_splitter.split_documents(documents)
        for idx, text in enumerate(self.docs):
            self.docs[idx].metadata['filename'] = self.file
            # self.docs[idx].metadata['filename'] = self.file.get('name')   
        filtered_docs = utils.filter_complex_metadata(self.docs)
        self.db = Chroma.from_documents(filtered_docs, embeddings)  

    def read_file(self, url):
        logger.info(url)
        loader = WebBaseLoader(url)
        self.documents = loader.load_and_split()  
        return self.documents    
    
# TODO - NOT IMPLEMENTED
class PandasWrapperHandler(Handler):
    def handle(self):
        return f"Wrapping {self.file} in Pandas dataframe"

    def _create_agent(self):
        self.df.columns = self.df.columns.str.strip()
        tools = load_tools(["python_repl"], llm=self.llm)
        self.agent = create_pandas_dataframe_agent(
            tools=tools, 
            llm=self.llm, 
            df=self.df, 
            verbose=True, 
            # These are the only two agents impelemnted for pandas at the moment
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            # max_execution_time=2,
            # early_stopping_method="generate",
            format_instructions = FORMAT_INSTRUCTIONS
            )

# TODO - NOT IMPLEMENTED
class ExcelHandler(PandasWrapperHandler):
    def handle(self):
        return f"Handling Excel file: {self.file}"
    
    def read_file(self, url, SLACK_BOT_TOKEN): 
        file_content = self._read_file_content(url, SLACK_BOT_TOKEN)
        self.df = pd.read_excel(file_content, sheet_name=0)                   
        # df.columns = df.columns.str.strip()
        # tools = load_tools(["llm-math"], llm=self.llm)
        # self.agent = create_pandas_dataframe_agent(
        #     tools=tools, 
        #     llm=self.llm, 
        #     df=df, 
        #     verbose=True, 
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     max_execution_time=1,
        #     early_stopping_method="generate"
        #     )
        self._create_agent()
        return self.q_and_a(self.first_impression)
        
# TODO - NOT IMPLEMENTED
class CSVHandler(PandasWrapperHandler):
    def handle(self):
        return f"Handling CSV file: {self.file}"

    def read_file(self, url, SLACK_BOT_TOKEN):  
        file_content = self._read_file_content(url, SLACK_BOT_TOKEN).decode('utf-8')
        self.df = pd.read_csv(StringIO(file_content))     
        # df.columns = df.columns.str.strip()
        # tools = load_tools(["llm-math"], llm=self.llm)        
        # self.agent = create_pandas_dataframe_agent(tools=tools, llm=self.llm, df=df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        # # print(agent.agent.llm_chain.prompt.template)
        # # return (agent.run(self.first_impression))     
        self._create_agent()
        return self.q_and_a(self.first_impression)
            # self.agent.run("What's the average rate of change from month to month?"))

# TODO - NOT IMPLEMENTED
class JSONHandler(Handler):
    def handle(self):
        return f"Handling JSON file: {self.file}"   
    
    def read_file(self, url, SLACK_BOT_TOKEN):
        file_content = self._read_file_content(url, SLACK_BOT_TOKEN)
        str_content = file_content.decode('utf-8')
        df = pd.read_json(StringIO(str_content))
        df.columns = df.columns.str.strip()
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=self.openai_api_key), df=df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        # print(agent.agent.llm_chain.prompt.template)


class MarkdownHandler(Handler):
    def handle(self):
        return f"Handling Markdown file: {self.file}"  

    def read_file(self):
        pass    

class HandlerFactory:
    # Note - WebHandler isn't in here because this factory is based on 
    # filetype. Instead, we instantiate it directly if we know we're
    # dealing with a web page
    handlers = {
        "pdf": PDFHandler, 
        "docx": DOCXHandler, 
        "txt": TxtHandler,
        # "xlsx": ExcelHandler, 
        # "json": JSONHandler,
        # "md": MarkdownHandler,
        # "csv": CSVHandler,
        }

    @classmethod
    def get_handler(cls, file, open_api_key, slack_bot_token):
        # _, file_extension = os.path.splitext(file)
        _, file_extension = os.path.splitext(file.get('name'))
        file_type = file_extension.lstrip('.').lower()
        Handler = cls.handlers.get(file_type)
        if Handler is None:
            raise ValueError(f"No handler for file type {file_type}")
        return Handler(file, open_api_key, slack_bot_token)

def create_file_handler(file, openai_api_key, slack_bot_token, webpage=False):
    # is_url = 'http://' in file.get('name') or 'https://' in file.get('name')
    is_url = 'http://' in file or 'https://' in file
    if is_url or webpage:
        # file = {'name': file, 'id': file, 'url_private': file}
        handler = WebHandler(file, openai_api_key, slack_bot_token)
    # elif webpage:
    #     handler = WebHandler(file, openai_api_key, slack_bot_token)
    else:
        handler = HandlerFactory.get_handler(file, openai_api_key, slack_bot_token)
    return handler

class FileRegistry:
    def __init__(self):
        self.registry = {}

    # def add_file(self, filename, channel_id, thread_ts, file_id, private_url, handler, chatWithDoc):
    def add_file(self, filename, channel_id, thread_ts, file_id, url_private, handler):
        if filename not in self.registry:
            self.registry[filename] = {}
        key = (channel_id, thread_ts)
        if key not in self.registry[filename]:
            self.registry[filename][key] = []
        self.registry[filename][key].append(
                {'file_id': file_id, 
                 'private_url': url_private, 
                 'handler': handler, 
                #  'chat': chatWithDoc,
                 })

    def get_files(self, filename, channel_id, thread_ts):
        key = (channel_id, thread_ts)
        if filename in self.registry and key in self.registry[filename]:
            return self.registry[filename][key]
        else:
            return None

    def list_files(self, filename):
        if filename in self.registry:
            return [file for sublist in self.registry[filename].values() for file in sublist]
        else:
            return None
