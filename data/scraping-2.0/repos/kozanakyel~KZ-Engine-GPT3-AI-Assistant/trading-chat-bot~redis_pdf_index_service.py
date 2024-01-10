import openai
import os
import tiktoken
import textract
from dotenv import load_dotenv
from tqdm import tqdm

from database import get_redis_connection
from database import get_redis_results

# Setup Redis
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

from config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME

from transformers import handle_file_string

load_dotenv()
openai_api_key = os.getenv('T_OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

# Constants
VECTOR_FIELD_NAME = 'content_vector'
VECTOR_DIM = 1536 #len(data['title_vector'][0]) # length of the vectors
#VECTOR_NUMBER = len(data)                 # initial number of vectors
PREFIX = "gptversedoc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
# Index
INDEX_NAME = "f1-index"           # name of the search index

class IndexRedisService:
    
    def __init__(self):
        self.redis_client = get_redis_connection()
        filename = TextField("filename")
        text_chunk = TextField("text_chunk")
        file_chunk_index = NumericField("file_chunk_index")

        # define RediSearch vector fields to use HNSW index

        text_embedding = VectorField(VECTOR_FIELD_NAME,
            "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": DISTANCE_METRIC
            }
        )
        # Add all our field objects to a list to be created as an index
        self.fields = [filename,text_chunk,file_chunk_index,text_embedding]
        openai.api_key = openai_api_key

    def index_checker(self):
        try:
            self.redis_client.ft(INDEX_NAME).info()
            print("Index already exists")
        except Exception as e:
            print(e)
            # Create RediSearch Index
            print('Not there yet. Creating')
            self.redis_client.ft(INDEX_NAME).create_index(
                fields = self.fields,
                definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
            )
            
    def initiliaze_tokenizer(self):
        openai.api_key = openai_api_key
        # Initialise tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Process each PDF file and prepare for embedding
        pdf_files = self.get_pdf_files()
        for pdf_file in tqdm(pdf_files):
    
            pdf_path = os.path.join(self.get_data_dir(), pdf_file)
            print(pdf_path)
    
            # Extract the raw text from each PDF using textract
            text = textract.process(pdf_path, method='pdfminer')
    
            # Chunk each document, embed the contents and load to Redis
            handle_file_string((pdf_file, text.decode("utf-8")), tokenizer, self.redis_client, VECTOR_FIELD_NAME,INDEX_NAME)

    def get_data_dir(self):
        return os.path.join('/mnt/c/Users/kozan/Desktop/Sen_Des_Proj/GPT-4-KZEngine-Signal-Interpretation/trading-chat-bot','data')    # change to DATA_PATH pdfs folders
    
    def get_pdf_files(self):
        pdf_files = sorted([x for x in os.listdir(self.get_data_dir()) if 'DS_Store' not in x])
        return pdf_files
    
    def get_number_of_docs(self):
        return self.redis_client.ft(INDEX_NAME).info()['num_docs']
    
    def response_f1_query(self, f1_query):

        result_df = get_redis_results(self.redis_client, f1_query, index_name=INDEX_NAME)
        # Build a prompt to provide the original query, the result and ask to summarise for the user
        summary_prompt = '''Summarise this result like assitant of my AI project to answer the search query a customer has sent.
        Search query: SEARCH_QUERY_HERE
        Search result: SEARCH_RESULT_HERE
        Summary:
        '''
        summary_prepped = summary_prompt.replace('SEARCH_QUERY_HERE',f1_query).replace('SEARCH_RESULT_HERE',result_df['result'][0])
        summary = openai.Completion.create(engine=COMPLETIONS_MODEL,prompt=summary_prepped,max_tokens=200)
        # Response provided by GPT-3
        # print(summary['choices'][0]['text'])
        return summary['choices'][0]['text']
    
    
if __name__ == '__main__':
    redis_service = IndexRedisService()
    pdf_files = redis_service.get_pdf_files()
    redis_service.index_checker()
    redis_service.initiliaze_tokenizer()
    response_f1 = redis_service.response_f1_query("what are the motivation concept for kzengine?")  
    print(f'response from our service: {response_f1} and type {type(response_f1)}')
    
