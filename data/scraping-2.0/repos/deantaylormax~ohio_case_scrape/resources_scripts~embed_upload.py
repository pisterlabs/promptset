# %%
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm  # this is our progress bar
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()
# Set up the Pinecone vector database
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1-aws"
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
# docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name) #this uploads embeddings and other data to Pinecone
index_name = "doc-prod-test"
index = pinecone.Index(index_name)
# Root directory
root_directory = '/Users/deantaylor/ohio_case_scrape'
# Path to checkpoint file
checkpoint_path = f'{root_directory}/checkpoint.txt'
district = 'District_10'

# %%
def meta_data_extractor(file_name):
    case_no = file_name.split('/')[-1].split('.')[0]
    district = file_name.split('/')[-3].split('_')[-1]
    year = int(file_name.split('/')[-2])
    url = f'https://www.supremecourt.ohio.gov/rod/docs/pdf/{district}/{year}/{case_no}.pdf'
    return case_no, district, year, url

# %%
import os

def process_file(file_path):
    """
    Process the file. Replace this function with your actual file processing logic.
    """
    case_no, district, year, url = meta_data_extractor(file_path)
    # file_path = '/Users/deantaylor/ohio_case_scrape/District_12/2005/2005-Ohio-5048.pdf'
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 2000,
    chunk_overlap  = 200,
    )  #required for loading into the embeddings model because of the limited context window
    texts = text_splitter.split_documents(data)
    for t in texts:  #this cleans up the text and adds metadata while removing the reference to the source file on my system which is not needed
        #replace the \n characters with spaces
        t.page_content = t.page_content.replace('\n', ' ')
        t.metadata["case_no"] = case_no
        t.metadata["district"] = district
        t.metadata["year"] = int(year)
        t.metadata["url"] = url
        t.metadata.pop('source')
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) #creates the object to make the embeddings, does not hold any data itself
    test_cone = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    print(f"{file_path}: Processed")
    # Add your file processing code here

def update_checkpoint(checkpoint_path, last_processed_file):
    """
    Updates the checkpoint with the last processed file.
    """
    with open(checkpoint_path, 'w') as file:
        file.write(last_processed_file)

def get_last_processed_file(checkpoint_path):
    """
    Reads the last processed file from the checkpoint file.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            return file.read().strip()
    return None

def process_district_year_files(district):
    """
    Process files in each year directory for a specified district, resuming from the last checkpoint.
    """
    last_processed = get_last_processed_file(checkpoint_path)
    start_processing = False if last_processed else True

    district_dir = os.path.join(root_directory, district)
    if os.path.exists(district_dir):
        # Iterate through each year within the district
        for year in range(2003, 2024):
            year_dir = os.path.join(district_dir, str(year))
            if os.path.exists(year_dir):
                files = os.listdir(year_dir)
                # Sort files to process them in order
                files.sort()

                for file in files:
                    file_path = os.path.join(year_dir, file)
                    if os.path.isfile(file_path):
                        # if year_dir == '/Users/deantaylor/ohio_case_scrape/District_4/2023':
                        #     print(file_path)
                        # Start processing if last_processed is None or we reached the last processed file
                        if start_processing or file_path == last_processed:
                            start_processing = True
                            process_file(file_path)
                            update_checkpoint(checkpoint_path, file_path)

process_district_year_files(district)