import openai
import os
import pinecone
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext, GPTVectorStoreIndex, set_global_service_context, download_loader
from llama_index.llms import OpenAI
from memory import ChatMemoryBuffer
from llama_index.vector_stores import PineconeVectorStore

# ==================== START SETUP LOGGER ====================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("Pinecone")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

logger.debug("Testing debug message")
logger.info("Testing info message")
logger.warning("Testing warning message")
logger.error("Testing error message")
logger.critical("Testing critical message")
# ==================== FINISH SETUP LOGGER ====================



# ==================== START SETUP ENV ====================
# pinecone.create_index("pdf-index", dimension=1536, metric="cosine", pod_type="p1")
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0))
set_global_service_context(service_context)
# ==================== FINISH SETUP ENV ====================



# ==================== MAIN ====================
index_name = "pdf-index"

pinecone_index = pinecone.Index(index_name)
logger.debug("Successfully get pinecone index.")

storage_context = StorageContext.from_defaults(vector_store=PineconeVectorStore(pinecone_index=pinecone_index))
logger.debug("Successfully get storage context.")

# documents = download_loader("PDFReader")().load_data(file=Path("file1.pdf"))
# logger.debug("Successfully get documents.")
# index = GPTVectorStoreIndex.from_documents(documents, )# storage_context=storage_context) # 여기 주석을 없애면, 파일 추가 학습 가능
metadata_filters = {"wiki_title": "file1.pdf"},
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, metadata_filters=metadata_filters)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
response = pinecone_index.as_query_engine(service_context=service_context).query("분쟁조정사건 유형에 대해 알려 줘.")
print(response)

'''
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="You are a chatbot, able to have normal interactions.",
    temperature=0,
)
logger.debug("Successfully get chat engine.")

query = "법 제2조 제5호에 대해 알려 줘."
response = chat_engine.stream_chat(f"학습한 file1.pdf에서 대답해 줘: 처리방침에서, {query}. 한국어로 대답해 줘.")
logger.debug("Successfully get response from chat engine.")

for token in response.response_gen:
    print(token, end="", flush=True)
    time.sleep(0.001)
'''