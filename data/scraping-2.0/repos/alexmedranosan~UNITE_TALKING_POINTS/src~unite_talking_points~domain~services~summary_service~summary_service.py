from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.unite_talking_points.domain.entities.entities import Document
from src.unite_talking_points.domain.services.service import Service


class SummaryService(Service):
    """
    A service that summaries text documents using LangChain and OpenAI models.
    """

    def __init__(self, document: Document, openai_api_key: str):
        """
        A service that summarizes text documents using LangChain and OpenAI models.
        :param document: Document The Document to be summarized.
        :param openai_api_key: str The OpenAI API key.
        """
        super().__init__()
        self.document = document
        self.openai_api_key = openai_api_key
        self.summary = None
        self.chunks = None
        self.chain = None
        self.summary = ''

    def _pre_process(self):
        """
        Pre-processes the document.

        This includes initializing the model connection, splitting the document into chunks,
        and loading the summarization chain.
        """
        # Initialize the model connection
        llm = OpenAI(temperature=0., openai_api_key=self.openai_api_key)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        self.chunks = text_splitter.create_documents([self.document.content])

        # Load the summary chain
        self.chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)

    def _process(self):
        """
        Processes the document.

        This includes running the summarization chain on the chunks.
        """
        self.summary = self.chain.run(self.chunks)

    def _post_process(self):
        """
        Post-processes the document.

        Returns the summarized document.
        """
        output = self.summary

        return output
