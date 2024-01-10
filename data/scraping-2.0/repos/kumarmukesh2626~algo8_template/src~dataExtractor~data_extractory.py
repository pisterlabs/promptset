import os
import re
import logging
from cgitb import text
import textract
import docx
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from dotenv import load_dotenv
from database.ConnectDb import DatabaseHandler
from PyPDF2 import PdfReader
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings, CohereEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from transformers import pipeline
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
openai.api_key = os.getenv("openai_api_key")
class DataExtractor:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.db = DatabaseHandler()

    def read_data(self, file_path_or_url):
        # Read data from various file formats based on file extension or download from URL
        # Return the ingested data
        data = None
        # Add code to read data from file_path_or_url based on file extension or download from URL
        try:
            if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
                # Download the file from the URL
                response = requests.get(file_path_or_url)
                file_extension = os.path.splitext(file_path_or_url)[1].lower()
                temp_file_path = "/path/to/temp/file" + file_extension  # Specify the temporary file path
                with open(temp_file_path, "wb") as file:
                    file.write(response.content)
                file_path = temp_file_path
            else:
                file_path = file_path_or_url

            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == ".csv":
                data = pd.read_csv(file_path)
            elif file_extension == ".xlsx" or file_extension == ".xls":
                data = pd.read_excel(file_path)
            elif file_extension == ".json":
                data = pd.read_json(file_path)
            elif file_extension == ".pdf":
                text = textract.process(file_path, method='tesseract', encoding='utf-8')
                data = text.decode('utf-8')
            elif file_extension == ".xml":
                # Handle XML files
                tree = ET.parse(file_path)
                root = tree.getroot()
                # Add code to extract data from XML
                pass
            elif file_extension == ".doc" or file_extension == ".docx":
                # Handle DOC files
                doc = docx.Document(file_path)
                # Add code to extract data from DOC
                pass
            else:
                logging.error("Unsupported file type")
        except Exception as e:
            logging.error(f"Error reading data from file or URL: {e}")
        return data

    def preprocess_data(self, data):
        # Perform data cleaning and preprocessing using regex
        # Return the preprocessed data
        preprocessed_data = None
        # Add code to preprocess the data
        try:
            if data is not None:
                if isinstance(data, str):
                    # Example: Remove special characters and digits
                    preprocessed_data = re.sub(r'[^a-zA-Z\s]', '', data)
                else:
                    preprocessed_data = data.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
        return preprocessed_data
    
    def text_chunks(self, raw_text):
        #st.write(raw_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        chunks = text_splitter.split_text(raw_text)
        docs = text_splitter.create_documents(chunks)

        return docs

    def generate_embeddings(self, data, embeddings_choice):
        # Generate embeddings for the preprocessed data using the specified embeddings choice
        # Return the embeddings
        # Add code to generate embeddings
        try:
            if data is not None:
                if embeddings_choice == "openai_text-embedding-ada-002":
                    # Initialize OpenAI Text Embedding Ada model
                    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("embeddings_key"))
                elif embeddings_choice == "huggingface_sentence-transformers/all-mpnet-base-v2":
                    # Initialize Hugging Face Sentence Transformers model
                    embeddings = HuggingFaceHubEmbeddings()
                elif embeddings_choice == "huggingface_hkunlp/instructor-large":
                    # Initialize Hugging Face HKUNLP Instructor model
                    embeddings = HuggingFaceInstructEmbeddings()
                elif embeddings_choice == "cohere_medium":
                    # Initialize Cohere Medium model
                    embeddings = CohereEmbeddings(cohere_api_key=os.getenv("embeddings_key"))
                else:
                    raise ValueError("Invalid embeddings choice")
                return embeddings
        except Exception as e:
            logging.error(f"Error occurred during embeddings generation: {e}")
            return None

    def store_embeddings(self, vectorstore, embeddings):
        try:
            # Store the embeddings in the specified vector store
            docsearch = None  # Initialize the docsearch variable
            if embeddings is not None:
                db = FAISS.from_documents(vectorstore, embeddings)
                db.save_local("faiss_index")
            else:
                print("Embeddings cannot be None")
                logging.error("Embeddings cannot be None")
            return docsearch
        except Exception as e:
            logging.error(f"Error occurred while storing embeddings: {e}")
            return None

    def generate_prompt(self, fields, data):
        try:
            # Create a prompt by combining the desired fields with the data
            prompt = f"Extract the following fields from the given data:\n"
            for field in fields:
                prompt += f"- {field}\n"
            prompt += f"\nData:\n{data}"
            return prompt
        except Exception as e:
            logging.error(f"Error occurred during prompt generation: {e}")
            return None
        
    def count_tokens(self, string):
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        return len(tokenizer(string)['input_ids'])
    
    def get_completion(self,text,prompt, model="gpt-3.5-turbo"):
        text = text[:4090] + "..." if len(text) > 4090 else text

        prompt = f""" Please follow the instruction given in  {prompt} and give your response below for following give context:
        ```{text}```
        """
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]
    def model_inference(self, embeddings, prompt,model_name):
        try:
            # Initialize the model based on the provided model name
            if model_name == "gpt-3.5-turbo":
                # Initialize OpenAI GPT-3.5 Turbo model
               response =  self.get_completion(embeddings,prompt,model_name)
            elif model_name == "llama2":
                # Initialize Llama2 model
                model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B-llama")
            elif model_name == "mistral":
                # Initialize Mistral model
                model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B-mistral")
            elif model_name == "orca2":
                # Initialize Orca2 model
                model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B-orca")
            else:
                logging.error("Invalid model name")
                return None
            return response
        except Exception as e:
            logging.error(f"Error occurred during model inference: {e}")
            return None

    def main(self):
        file_path = os.getenv("FILE_PATH2")
        fields_to_extract = os.getenv("FIELDS_TO_EXTRACT")
        # fields_to_extract = fields_to_extract.split(",")

        # Data Ingestion
        logging.info("Reading data from file...")
        print(file_path)
        data = self.read_data(file_path)
        # Data Preprocessing
        logging.info("Performing data preprocessing...")
        preprocessed_data = self.preprocess_data(data)
        text_chunks = self.text_chunks(preprocessed_data)
        # Embeddings
        logging.info("Generating embeddings...")
        embeddings = self.generate_embeddings(text_chunks, os.getenv("EMBEDDINGS_CHOICE"))
        # Vector Store
        logging.info("Storing embeddings...")
        vectorStore = self.store_embeddings(text_chunks,embeddings)
        # Prompt Generation
        logging.info("Generating prompt...")
        prompt = self.generate_prompt(fields_to_extract, text_chunks)

        # LLM Model Inference
        logging.info("Performing model inference...")
        response = self.model_inference(text_chunks,prompt,os.getenv("LLM_MODEL_NAME"))
        print(response)
        # Database Integration
        logging.info("Pushing data to database...")
        # self.push_to_database(respone)

        # Count Tokens
        logging.info("Counting tokens...")
        token_count = self.count_tokens(response)
        print(f"Token count: {token_count}")

if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.main()
