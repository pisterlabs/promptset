import os
import shutil
import urllib

from langchain.text_splitter import CharacterTextSplitter

from autodev.qa.document_db import SingleTextFileDocumentDatabase
from autodev.llm import LLMType
from autodev.util import logging
from autodev.qa.qa_use_case import QuestionAnsweringUseCase

log = logging.getLogger(__name__)


class DocumentDatabaseFirefaceManual(SingleTextFileDocumentDatabase):
    def __init__(self):
        manual_path = os.path.join('data', 'fface_uc_e.txt')
        if not os.path.exists(manual_path):
            log.info("Fireface manual not found. Downloading...")
            url = "https://drive.google.com/uc?export=download&id=1cx2kljZY-CNQa3jdq4oGqpXYNQ3nFJDG"
            with urllib.request.urlopen(url) as response, open(manual_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        super().__init__("fireface", manual_path)


class UseCaseFirefaceManual(QuestionAnsweringUseCase):
    def __init__(self, llm_type: LLMType):
        queries = [
            "What is the impedance of the instrument input?",
            "What is the purpose of the matrix mixer?",
            "How many mic inputs does the device have?",
            "Can the device be powered via USB alone?",
            "What is the minimum round trip latency?",
            "How can I invert stereo for an output channel?",
            "How can I change the headphone output volume on the device itself?",
            "What is a submix and how many submixes can be defined?",
        ]
        super().__init__(llm_type=llm_type, doc_db=DocumentDatabaseFirefaceManual(),
            splitter=CharacterTextSplitter(chunk_size=llm_type.chunk_size(), chunk_overlap=0),
            queries=queries)


if __name__ == '__main__':
    logging.configure()

    use_case = UseCaseFirefaceManual(LLMType.OPENAI_DAVINCI3)
    #use_case = UseCaseFirefaceManual(LLMType.OPENAI_CHAT_GPT4)
    #use_case = UseCaseFirefaceManual(LLMType.HUGGINGFACE_GPT2)

    use_case.run_example_queries()

