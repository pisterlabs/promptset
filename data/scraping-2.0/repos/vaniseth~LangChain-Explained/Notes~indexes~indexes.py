from langchain.document_loaders.notion import NotionDirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.email import UnStructuredEmailLoader
import dotenv

from dotenv import load_dotenv  
load_dotenv()

loader = NotionDirectoryLoader('Notion_DB')
loader = CSVLoader ('testing.csv')
loader = PyPDFLoader ('testing.pdf')
loader = UnStructuredEmailLoader('testing-email.eml')

data = loader.load()