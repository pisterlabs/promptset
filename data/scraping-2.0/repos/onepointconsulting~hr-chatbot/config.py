from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    faiss_persist_directory_uk = Path(os.environ["FAISS_STORE_UK"])
    faiss_persist_directory_india = Path(os.environ["FAISS_STORE_INDIA"])
    faiss_dirs = [faiss_persist_directory_uk, faiss_persist_directory_india]
    for d in faiss_dirs:
        if not d.exists():
            d.mkdir()

    doc_location_uk = Path(os.environ["DOC_LOCATION_UK"])
    doc_location_india = Path(os.environ["DOC_LOCATION_INDIA"])
    doc_locations = [doc_location_uk, doc_location_india]

    location_persistence_map = {
        "GB": {
            "faiss_persist_directory": faiss_persist_directory_uk,
            "doc_location": doc_location_uk,
        },
        "IN": {
            "faiss_persist_directory": faiss_persist_directory_india,
            "doc_location": doc_location_india,
        },
    }

    for location in doc_locations:
        if not location.exists():
            raise Exception(f"File not found: {location}")

    embeddings = OpenAIEmbeddings(chunk_size=100)
    model = "gpt-3.5-turbo-16k"
    # model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0)
    search_results = 5

    def __repr__(self) -> str:
        return f"""# Configuration
faiss_persist_directories: {self.faiss_dirs}
doc_locations:             {self.doc_locations}

embeddings: {self.embeddings}

llm: {self.llm}
"""


cfg = Config()

if __name__ == "__main__":
    print(cfg)
