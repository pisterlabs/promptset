from langchain.chat_models import ChatOpenAI
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    model = "gpt-3.5-turbo-0613"
    # model = 'gpt-4-0613'
    llm = ChatOpenAI(model=model, temperature=0)
    doc_location = Path(os.getenv("DOC_LOCATION"))
    test_doc_location = Path(os.getenv("TEST_DOCS"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    temp_doc_location = Path(os.getenv("TEMP_DOC_LOCATION"))

    if not temp_doc_location.exists():
        temp_doc_location.mkdir(parents=True)

    def __repr__(self) -> str:
        return f"""# Configuration

doc_location: {self.doc_location}
llm: {self.llm}
"""


cfg = Config()

if __name__ == "__main__":
    print(cfg)
