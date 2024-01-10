import ast
from importlib import metadata
from flask import Blueprint, jsonify, request
from flask_cors import CORS
import os

from typing import Any
from langchain.chains import RetrievalQAWithSourcesChain
import langchain
from numpy import source
from blueprints.dto.documents import Documents
from blueprints.process.models.model_openai import MODEL_OPENAI

from blueprints.process.process_chunks import PROCESS_CHUNKS
from blueprints.process.process_scrape import SCRAPER
from blueprints.process.process_vectorstore import PROCESS_VECTORSTORE


class ProcessView:
    """
    A class representing the process view.

    This class handles the processing of URLs.

    Attributes:
        bp (Blueprint): The Flask Blueprint object for the process URLs.
        process_urls_model (object): The model for processing URLs.

    Methods:
        __init__(): Initializes the ProcessView object.
        add_cors_headers(response): Adds CORS headers to the response.
        process_urls(): Processes the URLs and returns the result.
    """

    def __init__(self) -> None:
        """
        Initializes the ProcessView object.

        This method sets up the Flask Blueprint, adds CORS headers,
        and defines the URL rule for processing URLs.
        """
        self.bp = Blueprint('process', __name__, url_prefix=os.environ['APP_PREFIX_ENDPOINT'] + '/process')

        CORS(self.bp)
        
        self.process_scraper_model = SCRAPER
        self.process_chunk_model = PROCESS_CHUNKS
        self.process_vectorstore_model = PROCESS_VECTORSTORE
        self.model_openai = MODEL_OPENAI()

        self.bp.after_request(self.add_cors_headers)

        self.bp.add_url_rule('/urls/', view_func=self._process_scraper, methods=['POST'])
        self.bp.add_url_rule('/chunks/', view_func=self._process_chunks, methods=['POST'])
        self.bp.add_url_rule('/question_answer_chain/', view_func=self._process_question_answer, methods=['POST'])
        self.bp.add_url_rule('/question_answer_withqa/', view_func=self._process_question_answer_withQA, methods=['POST'])
        
    def add_cors_headers(self, response) -> Any:
        """
        Adds CORS headers to the response.

        Args:
            response (Any): The Flask response object.

        Returns:
            Any: The modified response object with CORS headers added.
        """
        response.headers["Access-Control-Allow-Origin"] = "*" 
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "Accept, Content-Type, Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
    
    def _process_chunks(self) -> Any:
        """
        Processes the URLs and returns the result.

        Returns:
            Any: The result of processing the URLs.
        """
        try:
            data = request.get_json() # type: ignore
            
            docs = data.get('docs', [])

            return self.process_chunk_model.process_chunks(docs)

        except Exception as e:
            return jsonify({'error': str(e)})
    
    def _process_scraper(self) -> Any:
        """
        Processes the URLs and returns the result.

        Returns:
            Any: The result of processing the URLs.
        """
        try:
            data = request.get_json() # type: ignore

            urls = data.get('urls', [])

            docs = self.process_scraper_model.start_scrape(urls)

            return docs

        except Exception as e:
            return jsonify({'error': str(e)})
        
    def _process_question_answer(self) -> Any:
        
        try:
            data = request.get_json() # type: ignore

            question = data.get('question', "")
            chunks_doc = data.get('chunks_docs', [])

            chunks_docs = [Documents().from_dict(doc) for doc in chunks_doc]
            
            pinecone_index = self.process_vectorstore_model.index(chunks_docs, self.model_openai.MODEL_EMBEDDINGS) # type: ignore
            
            retrieve_relevant_chunks = self.process_vectorstore_model.retrieve_from_db(pinecone_index, question)
            
            sources = " ".join(i for i in set([s.metadata['source'] for s in retrieve_relevant_chunks])) # type: ignore
                        
            openai_chain = self.model_openai.chain()
            response = openai_chain.run(input_documents=retrieve_relevant_chunks, question=question)

            return jsonify({
                "answer" : response,
                "question" : question,
                "source": sources
            })

        except Exception as e:
            return jsonify({'error': str(e)})
        
    def _process_question_answer_withQA(self) -> Any:
        try:
            data = request.get_json() # type: ignore

            question = data.get('question', '')
            chunks_doc = data.get('chunks_docs', [])

            langchain.debug = False # type: ignore

            chunks_docs = [Documents().from_dict(doc) for doc in chunks_doc]
            
            pinecone_index = self.process_vectorstore_model.index(chunks_docs, self.model_openai.MODEL_EMBEDDINGS)
            
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=self.model_openai.openai_llm, 
                retriever=pinecone_index.as_retriever())

            response = chain({"question":question})
            
            return response

        except Exception as e:
            return jsonify({'error': str(e)})
        
