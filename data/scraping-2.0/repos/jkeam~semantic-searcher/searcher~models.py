from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.llms import OpenAI
from chromadb import HttpClient
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai.error import AuthenticationError, RateLimitError
from logging import getLogger
from searcher.extensions import db
from datetime import datetime

class Fact(db.Model):
    id = db.Column(db.String(40), primary_key=True)
    title = db.Column(db.String(256))
    body = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    author_id = db.Column(db.Integer)

    def __repr__(self):
        return f'<Fact "{self.title}">'

class TrainingError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

class Searcher:
    def __init__(self, openai_api_key:str, open_ai_model_name:str, chroma_host:str, chroma_port:str):
        openai:OpenAI = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self._chain:BaseCombineDocumentsChain = load_qa_chain(openai, chain_type='stuff')
        self._dbclient = HttpClient(host=chroma_host, port=chroma_port)
        self._collection_name = "chroma"
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=open_ai_model_name
        )
        self._collection = self._dbclient.get_or_create_collection(name=self._collection_name)
        self._logger = getLogger(__name__)


    def train(self, values):
        """
        Train the model
        """
        doc_str = "\n\n".join(values)
        self._collection = self._generate_index(doc_str)


    def _generate_index(self, text:str) -> Collection:
        """
        Index the document and return the indexed db
        """
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.split_text(text)

        self._dbclient.get_or_create_collection(name=self._collection_name)
        self._dbclient.delete_collection(name=self._collection_name)

        collection = self._dbclient.create_collection(name=self._collection_name, embedding_function=self._embedding_function)
        try:
            collection.add(documents=documents, ids=list(map(lambda num: str(num), range(len(documents)))))
        except AuthenticationError as e:
            self._logger.error(e)
            raise TrainingError('Invalid OPENAI Key')
        except RateLimitError as e:
            self._logger.error(e)
            raise TrainingError('Rate Limit Error while using OPENAI Key')
        return collection


    def _answer_question(self, query:str, collection:Collection, chain:BaseCombineDocumentsChain) -> str:
        """
        Takes in query, index to search from, and llm chain to generate answer
        """
        query_db = Chroma(client=self._dbclient, collection_name=self._collection_name, embedding_function=OpenAIEmbeddings())
        docs = query_db.similarity_search(query)
        answer:dict[str, str] = chain({'input_documents': docs, 'question': query}, return_only_outputs=True)
        return answer['output_text']


    def ask(self, query:str) -> str:
        """
        Ask the model a query
        """
        return self._answer_question(query, self._collection, self._chain)
