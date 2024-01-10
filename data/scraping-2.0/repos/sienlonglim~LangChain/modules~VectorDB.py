from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import get_openai_callback
import logging
logger = logging.getLogger(__name__)

class VectorDB():
    def __init__(self, config):
        self.config = config
        self.db_option = config['embedding_options']['db_option']
        self.document_names = None
        logger.info('VectorDB instance created')

    def create_embedding_function(self, openai_api_key : str):
        logger.info('Creating embedding function')
        self.embedding_function = OpenAIEmbeddings(
            deployment="SL-document_embedder",
            model=self.config['embedding_options']['model'],
            show_progress_bar=True,
            openai_api_key = openai_api_key) 
    
    def initialize_database(self, document_chunks : list, document_names : list):
        # Track tokem usage
        logger.info('Initializing vector_db')
        if self.db_option == 'FAISS':
            logger.info('\tRunning in memory')
            self.vector_db = FAISS.from_documents(
                documents = document_chunks, 
                embedding = self.embedding_function
                )
        logger.info('\tCompleted\n\n')
        self.document_names = document_names

    def create_llm(self, openai_api_key : str, temperature : int):
        # Instantiate the llm object 
        logger.info('Instantiating the llm')
        self.llm = ChatOpenAI(
            model_name=self.config['llm'],
            temperature=temperature,
            api_key=openai_api_key
            )
        

    def create_chain(self, prompt_mode : str, source : str):
        logger.info('Creating chain from template')
        if prompt_mode == 'Restricted':
            # Build prompt template
            template = """Use the following pieces of context to answer the question at the end. \
            If you don't know the answer, just say that you don't know, don't try to make up an answer. \
            Keep the answer as concise as possible. 
            Context: {context}
            Question: {question}
            Helpful Answer:"""
            qa_chain_prompt = PromptTemplate.from_template(template)
        elif prompt_mode == 'Unrestricted':
            # Build prompt template
            template = """Use the following pieces of context to answer the question at the end. \
            If you don't know the answer, you may make inferences, but make it clear in your answer. \
            Elaborate on your answer.
            Context: {context}
            Question: {question}
            Helpful Answer:"""
            qa_chain_prompt = PromptTemplate.from_template(template)

        # Build QuestionAnswer chain
        if source == 'Uploaded documents / weblinks':
            self.qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.vector_db.as_retriever(
                    search_type="similarity", # mmr, similarity_score_threshold, similarity
                    search_kwargs = {
                        "k": 3,                     #k: Amount of documents to return (Default: 4)
                        'score_threshold': 0.5,     # Minimum relevance threshold for similarity_score_threshold
                        'fetch_k': 5,              # Amount of documents to pass to MMR algorithm (Default: 20)
                        'lambda_mult': 0.5,         # Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)
                        }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_chain_prompt}
                )
        elif source == 'Wikipedia':
            wiki_retriever = WikipediaRetriever(
                top_k_results = 5,
                lang= "en", 
                load_all_available_meta = False,
                doc_content_chars_max = 4000,
                features="lxml"
                )
            self.qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=wiki_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_chain_prompt}
                )
            
        logger.info('\tCompleted')
    
    def get_response(self, user_input : str):
        # Query and Response
        logger.info('Getting response from server')
        with get_openai_callback() as cb:
            result = self.qa_chain({"query": user_input})
            logger.info(f"\n{cb}")
        logger.info('\tCompleted\n\n')
        return result


