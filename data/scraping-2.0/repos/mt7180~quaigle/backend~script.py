from llama_index import (
    SimpleWebPageReader,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
    get_response_synthesizer,
)
from llama_index.readers import BeautifulSoupWebReader

from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser.extractors import (
    MetadataExtractor,
)
from llama_index.node_parser.extractors.marvin_metadata_extractor import (
    MarvinMetadataExtractor,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.memory import ChatMemoryBuffer

from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

from marvin import ai_model
from llama_index.bridge.pydantic import BaseModel as LlamaBaseModel
from llama_index.bridge.pydantic import Field as LlamaField
import pathlib
import tiktoken
import logging
import os

from .document_categories import CATEGORY_LABELS


class AITextDocument:
    """Loads and converts a text file into LlamaIndex nodes.
    The marvin ai_model predicts the text category based on a
    given list and gives a short summary of the text based on the given llm_str.
    """

    cfd = pathlib.Path(__file__).parent / "data"

    def __init__(
        self,
        document_name: str,
        llm_str: str,
        callback_manager: CallbackManager | None = None,
    ):
        self.callback_manager: CallbackManager | None = callback_manager
        self.document = self._load_document(document_name)
        self.nodes = self.split_document_and_extract_metadata(llm_str)
        self.category = self.nodes[0].metadata["marvin_metadata"].get("category")
        text_subject = self.nodes[0].metadata["marvin_metadata"].get("description")
        self.summary = f'You uploaded a {self.category.lower()} text, please ask any \
            question about "{text_subject}".'

    @classmethod
    def _load_document(cls, identifier: str):
        """loads only the data of the specified name

        identifier: name of the text file as str
        """
        return SimpleDirectoryReader(
            input_files=[str(AITextDocument.cfd / identifier)],
            encoding="utf-8",
        ).load_data()[0]

    def _get_text_splitter(self):
        return TokenTextSplitter(
            separator=" ",
            chunk_size=1024,
            chunk_overlap=128,
            callback_manager=self.callback_manager,
        )

    def _get_metadata_extractor(self, llm_str):
        return MetadataExtractor(
            extractors=[
                MarvinMetadataExtractor(
                    marvin_model=AIMarvinDocument,
                    llm_model_string=llm_str,
                    show_progress=True,
                    callback_manager=self.callback_manager,
                ),
            ],
        )

    def split_document_and_extract_metadata(self, llm_str):
        # text_splitter = self._get_text_splitter()
        metadata_extractor = self._get_metadata_extractor(llm_str)
        node_parser = SimpleNodeParser.from_defaults(
            # text_splitter=text_splitter,
            metadata_extractor=metadata_extractor,
            callback_manager=self.callback_manager,
        )
        return node_parser.get_nodes_from_documents([self.document], show_progress=True)


@ai_model
class AIMarvinDocument(LlamaBaseModel):
    # description: str = LlamaField(
    #     ...,
    #     description="""A brief summary of the main content of the
    #     document.
    #     """,
    # )
    description: str = LlamaField(
        ...,
        description="""main subject of the text, e.g. only the name of
        a person or technology.
        """,
    )
    category: str = LlamaField(
        ...,
        description=f"""best matching text category from the following list: 
        {str(CATEGORY_LABELS)}
        """,
    )


class AIPdfDocument(AITextDocument):
    @classmethod
    def _load_document(cls, identifier: str):
        # loader = PDFReader()
        # return loader.load_data(
        #     file=pathlib.Path(str(AITextDocument.cfd / identifier))
        # )[0]
        return SimpleDirectoryReader(
            input_files=[pathlib.Path(str(AITextDocument.cfd / identifier))]
        ).load_data()[0]


class AIHtmlDocument(AITextDocument):
    @classmethod
    def _load_document_simplewebpageReader(cls, identifier: str):
        """loads the data of a simple static website at a given url
        identifier: url of the html file as str
        """
        return SimpleWebPageReader(
            html_to_text=True,
        ).load_data(
            [identifier]
        )[0]

    @classmethod
    def _load_document_BeautifulSoupWebReader(cls, identifier: str):
        """loads the data of an html file at a given url
        identifier: url of the html file as str
        """
        # from llama_index import download_loader

        # BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

        # loader = BeautifulSoupWebReader()
        # return loader.load_data(urls=[identifier])[0]
        return BeautifulSoupWebReader().load_data(urls=[identifier])[0]

    @classmethod
    def _load_document(cls, identifier: str):
        """loads the data of an html file at a given url
        identifier: url of the html file as str
        """
        if "wikipedia" in identifier:
            return cls._load_document_BeautifulSoupWebReader(identifier)

        # It's not easy to scrape complex/ dynamic websites, this is a task for
        # itself and is not covered here.
        # Currently LlamaHub offers some different options for WebReaders, but
        # the ultimative cost-free Reader seems not to be availabe, yet.
        # Check and if availabe implement better ones in future or work on
        # configuring existing ones for specific tasks ...
        # return cls._load_document_simplewebpageReader(identifier)
        return cls._load_document_BeautifulSoupWebReader(identifier)


class CustomLlamaIndexChatEngineWrapper:
    """A LlamaIndex CondenseQuestionChatEngine with RetrieverQueryEngine"""

    system_prompt: str = """You are a chatbot that responds to all questions about 
    the given context. The user gives you instructions on which questions to answer. 
    When you write the answers, you need to make sure that the user's expectations are 
    met. Remember that you are an accurate and experienced writer 
    and you write unique answers. Don't add anything hallucinatory.
    Use friendly, easy-to-read language, and if it is a technical or scientific text, 
    please stay correct and focused.
    Responses should be no longer than 10 sentences, unless the user explicitly 
    specifies the number of sentences.
    """

    OPENAI_MODEL = "gpt-3.5-turbo-instruct"
    # OPENAI_MODEL = "text-davinci-003"
    cfd = pathlib.Path(__file__).parent

    def __init__(self, callback_manager=None):
        self.callback_manager = callback_manager
        self.data_category: str = ""  # default, if no document is loaded yet
        self.llm = OpenAI(
            model=CustomLlamaIndexChatEngineWrapper.OPENAI_MODEL,
            temperature=0,
            max_tokens=512,
        )
        self.service_context = self._create_service_context()
        set_global_service_context(self.service_context)
        self.documents = []
        storage_dir = CustomLlamaIndexChatEngineWrapper.cfd / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        # logging.info(f"storage dir exists: {os.path.exists(storage_dir)}")
        if any(
            pathlib.Path(CustomLlamaIndexChatEngineWrapper.cfd / "storage").iterdir()
        ):
            self.storage_context = StorageContext.from_defaults(
                persist_dir=str(CustomLlamaIndexChatEngineWrapper.cfd / "storage"),
            )
            self.vector_index = load_index_from_storage(
                storage_context=self.storage_context
            )
        else:
            logging.debug("creating new vec index")
            self.vector_index = self.create_vector_index()
        self.chat_engine = self.create_chat_engine()

    def _create_service_context(self):
        return ServiceContext.from_defaults(
            chunk_size=1024,
            chunk_overlap=152,
            llm=self.llm,
            system_prompt=CustomLlamaIndexChatEngineWrapper.system_prompt,
            callback_manager=self.callback_manager,
        )

    def add_document(self, document: AITextDocument) -> None:
        self.documents.append(document)
        self._add_to_vector_index(document.nodes)
        self.data_category = document.category
        self.vector_index.storage_context.persist(
            persist_dir=CustomLlamaIndexChatEngineWrapper.cfd / "storage"
        )

    def clear_data_storage(self) -> None:
        doc_ids = list(self.vector_index.ref_doc_info.keys())
        for doc_id in doc_ids:
            self.vector_index.delete_ref_doc(doc_id, delete_from_docstore=True)
        self.vector_index.storage_context.persist(
            persist_dir=CustomLlamaIndexChatEngineWrapper.cfd / "storage"
        )
        self.documents.clear()
        # data folder with filed is cleared in respective route in fastapi_app.py

    def create_vector_index(self):
        return VectorStoreIndex(
            [
                node for doc in self.documents for node in doc.nodes
            ],  # current use case: no docs availabe, so empty list []
            service_context=self.service_context,
        )

    def _add_to_vector_index(self, nodes):
        self.vector_index.insert_nodes(
            nodes,
        )

    def _create_vector_index_retriever(self):
        vector_store_info = VectorStoreInfo(
            content_info="content of uploaded text documents",
            metadata_info=[
                MetadataInfo(
                    name="category",
                    type="str",
                    description="""best matching text category (e.g. Technical, 
                        Biagraphy, Sience Fiction, ... )
                    """,
                ),
                MetadataInfo(
                    name="description",
                    type="str",
                    # description="a brief summary of the document content",
                    description="main document content in one word",
                ),
            ],
        )
        return VectorIndexRetriever(
            index=self.vector_index,
            vector_store_info=vector_store_info,
            similarity_top=10,
        )

    def create_chat_engine(self) -> CondenseQuestionChatEngine:
        vector_query_engine = RetrieverQueryEngine(
            retriever=self._create_vector_index_retriever(),
            response_synthesizer=get_response_synthesizer(),
            callback_manager=self.callback_manager,
        )
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=vector_query_engine,
            memory=ChatMemoryBuffer.from_defaults(token_limit=1500),
            verbose=True,
            callback_manager=self.callback_manager,
        )

    def clear_chat_history(self) -> str:
        self.chat_engine.reset()
        return "Chat history succesfully cleared"

    def update_temp(self, temperature):
        # see https://gpt-index.readthedocs.io/en/v0.8.34/examples/llm/XinferenceLocalDeployment.html
        self.vector_index.service_context.llm.__dict__.update(
            {"temperature": temperature}
        )

    def answer_question(self, question: str) -> str:
        return self.chat_engine.chat(question.prompt)


def set_up_text_chatbot():
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    callback_manager = CallbackManager([token_counter])

    return (
        CustomLlamaIndexChatEngineWrapper(callback_manager=callback_manager),
        callback_manager,
        token_counter,
    )


if __name__ == "__main__":
    import sys
    import certifi
    from dotenv import load_dotenv

    # workaround for mac to solve "SSL: CERTIFICATE_VERIFY_FAILED Error"
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()

    load_dotenv()

    # API_KEY = os.getenv('OPENAI_API_KEY')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger(__name__).addHandler(logging.StreamHandler(stream=sys.stdout))
    # openai_log = "debug"

    chat_engine, callback_manager, token_counter = set_up_text_chatbot()
    chat_engine.clear_data_storage()

    try:
        url = "https://medium.com/how-ai-built-this/zero-to-one-a-guide-to-building-a-first-pdf-chatbot-with-langchain-llamaindex-part-1-7d0e9c0d62f"
        # "https://en.wikipedia.org/wiki/Sandor_Szondi"
        document = AIHtmlDocument(url, "gpt-3.5-turbo", callback_manager)
        chat_engine.add_document(document)

    except Exception as e:
        print(f"ERROR while loading and adding document to vector index: {e.args}")
        exit()

    while True:
        question = input("Your Question: ")
        response = chat_engine.chat_engine.chat(question)
        print(f"Agent: {response}")
        logging.info(
            f"Number of used tokens: {token_counter.total_embedding_token_count}"
        )
