from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    chunk_size = 6000
    chunk_overlap = 100
    chunk_separator = "\n\n"
    faiss_persist_directory = Path(os.environ["FAISS_STORE"])
    if not faiss_persist_directory.exists():
        faiss_persist_directory.mkdir()
    embeddings = OpenAIEmbeddings(chunk_size=400)
    model = "gpt-3.5-turbo-16k"
    # model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0)
    chat_hist_location = os.environ["CHAT_HISTORY_LOCATION"]
    history_file = Path(f"{chat_hist_location}/chat_history.txt")
    # To overcome rate limiting erros, please use https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
    search_results = 5


cfg = Config()
