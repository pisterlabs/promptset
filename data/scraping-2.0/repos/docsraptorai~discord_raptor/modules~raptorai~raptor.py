import os
from dotenv import load_dotenv
import argparse
from modules import utils
import psycopg2
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.vector_stores import PGVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import download_loader
from llama_index import VectorStoreIndex
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import set_global_service_context
import tiktoken
import petname


# gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-vision-preview, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo
EMBED_MODEL = 'text-embedding-ada-002'
EMBED_DIMENSION = 1536
INDEX_SUFFIX = '-index'
INDEX_TABLE_PREFIX = 'data_'
LLM_MODEL = 'gpt-3.5-turbo'
SYSTEM_DB = 'postgres'
VECTOR_DB = 'vector_db'
DOCSRAPTORAI_DB = 'docsraptorai'
RAPTOR_DEFAULT_NAME = 'ace'
LOGGER_RAPTOR_ROOT = 'raptor'

COST_1000_EMBEDDINGS = 0.0001
COST_1000_PROMPT = 0.001
COST_1000_COMPLETION = 0.002


class RaptorAI():
    logger = utils.get_logger('docsraptorai')
    raptor_logger = utils.get_logger(LOGGER_RAPTOR_ROOT)
    db_system= SYSTEM_DB
    db_docsraptorai = DOCSRAPTORAI_DB
    db_vector = VECTOR_DB
    db_host = None
    db_password = None
    db_port = None
    db_user = None
    db_connect_system = None
    db_connect = None

    def __init__(self):
        self.logger.info('init')
        self.init_db_connections()


    def init_db_connections(self):
        self.logger.info('  Initialize Postgres')

        self.logger.info('    system db connection')
        self.db_host = os.getenv('DB_HOST')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_connect_system = psycopg2.connect(
            dbname=self.db_system,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect_system.autocommit = True
        self.init_db()
        self.logger.info('    docsraptorai db connection')
        self.db_connect = psycopg2.connect(
            dbname=self.db_docsraptorai,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect.autocommit = True

    def init_db(self):
        self.logger.info(f'    Checking DB {self.db_docsraptorai}')
        with self.db_connect_system.cursor() as c:
            c.execute(f'select exists(select datname from pg_catalog.pg_database where datname=\'{self.db_docsraptorai}\')')
            docsraptorai_db_exist = c.fetchone()[0]
            if not docsraptorai_db_exist:
                self.logger.info(f'    Creating DB {self.db_docsraptorai}')
                c.execute(f'CREATE DATABASE {self.db_docsraptorai}')
                self.init_docsraptorai_db()

        if os.getenv('DB_RESET_INDEX') == 'true':
            self.logger.info(f'    Droping DB {self.db_vector}')
            with self.db_connect_system.cursor() as c:
                c.execute(f'DROP DATABASE IF EXISTS {self.db_vector}')

        self.logger.info(f'    Checking DB {self.db_vector}')
        with self.db_connect_system.cursor() as c:
            c.execute(f'select exists(select datname from pg_catalog.pg_database where datname=\'{self.db_vector}\')')
            vector_db_exist = c.fetchone()[0]
            if not vector_db_exist:
                self.logger.info(f'    Creating DB {self.db_vector}')
                c.execute(f'CREATE DATABASE {self.db_vector}')

    def init_docsraptorai_db(self):
        self.logger.info('      init docsraptorai db')
        connect = psycopg2.connect(
            dbname=self.db_docsraptorai,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        connect.autocommit = True
        with connect.cursor() as c:
            self.logger.info('creating raptor table')
            c.execute('CREATE TABLE raptor (id SERIAL PRIMARY KEY, name VARCHAR(64));')

    def get_raptor(self, name):
        return Raptor(name, EMBED_MODEL, EMBED_DIMENSION, LLM_MODEL, self.db_vector, self.db_connect)

    async def list(self):
        self.logger.info('listing raptors')
        raptor_list = [RAPTOR_DEFAULT_NAME]
        with self.db_connect.cursor() as c:
            c.execute('SELECT name from raptor')
            rows = c.fetchall()
            self.logger.info(f'  select result: {rows}')
            for raptor_tuple in rows:
                raptor_list.append(raptor_tuple[0])

        return raptor_list

    async def feed(self, url: str):
        self.logger.info(f'feeding with: {url}')

        raptor = self.get_raptor(RAPTOR_DEFAULT_NAME)
        raptor.feed(url)
        return 'yumi'

    async def ask(self, question: str):
        self.logger.info(f'asking: {question}')
        raptor = self.get_raptor(RAPTOR_DEFAULT_NAME)
        response = raptor.query(question)
        self.logger.info(f'Response: {response}')
        if (response.response == 'Empty Response'):
            return 'Rrrr, feed me first'
        else:
            return response.response

    async def kill(self):
        self.logger.info('kill raptor')
        raptor = self.get_raptor(RAPTOR_DEFAULT_NAME)
        raptor.suicide()
        return 'Raptor hunted sir'

    async def hatch(self):
        self.logger.info('hatch a new raptor')
        name = petname.generate()
        self.get_raptor(name)
        self.logger.info(f'  name: {name}')
        with self.db_connect.cursor() as c:
            c.execute(f'INSERT INTO raptor (name) VALUES (\'{name}\')')
        return name

class Raptor():
    name = None
    embed_model_name = None
    embed_model_dimension = None
    embed_model = None
    model_name = None
    llm = None
    service_context = None
    token_counter = None
    callback_manager = None
    db_host = None
    db_password = None
    db_port = None
    db_user = None
    db_vector_name = None
    db_connect = None
    db_connect_index = None

    logger = None

    def __init__(self, name, embed_model_name, embed_model_dimension, model_name, db_vector_name, db_connect):
        self.logger = utils.get_logger_child(f'{LOGGER_RAPTOR_ROOT}.{name}')
        self.logger.info(f'init {name}')
        self.name = name
        self.embed_model_name = embed_model_name
        self.embed_model_dimension = embed_model_dimension
        self.model_name = model_name
        self.db_vector_name = db_vector_name
        self.db_connect = db_connect
        self.init_db()
        self.init_embeddings()
        self.init_llm()
        self.init_llm_counters()
        self.init_service_context()

    def init_db(self):
        self.logger.info(f'  index vector db connection: {self.db_vector_name}')
        self.db_host = os.getenv('DB_HOST')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_connect_index = psycopg2.connect(
            dbname=self.db_vector_name,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect_index.autocommit = True

    def init_embeddings(self):
        self.logger.info('  embeddings')
        self.embed_model = OpenAIEmbedding(model=self.embed_model_name)

    def init_llm(self):
        self.logger.info('  init LLM')
        self.llm = OpenAI(model=self.model_name)

    def init_llm_counters(self):
        self.logger.info('    init LLM counters')
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(self.model_name).encode
        )
        self.callback_manager = CallbackManager([self.token_counter])

    def init_service_context(self):
        self.logger.info('  Define Service Context')

        self.service_context = ServiceContext.from_defaults(
                llm=self.llm, callback_manager=self.callback_manager, embed_model=self.embed_model
        )
        # Cannot find how to pass query embeddings from service context without setting global service context
        self.logger.info('  Set global service context for query embeddings')
        set_global_service_context(self.service_context)

    def get_vector_store(self, index_name):
        self.logger.info(f'Get vector store: {index_name}, dimension: {str(self.embed_model_dimension)}')

        return PGVectorStore.from_params(
            database=self.db_vector_name,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
            table_name=index_name,
            embed_dim=self.embed_model_dimension,
        )

    def get_storage_context(self, vector_store):
        self.logger.info('Get storage context')

        return StorageContext.from_defaults(vector_store=vector_store)

    def get_index(self):
        self.logger.info(f'Load index from stored vectors')
        index_name = self.index_from_name()
        vector_store = self.get_vector_store(index_name)
        storage_context = self.get_storage_context(vector_store = vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    def index_documents(self, index_name, documents):
        self.logger.info(f'Index documents in index: {index_name}')
        for document in documents:
            # self.logger.info(f'document: {document}')
            self.logger.info(f'document id: {document.doc_id}')
            # self.logger.info(f'extra info: {document.extra_info}')
        vector_store = self.get_vector_store(index_name)
        # self.logger.info(f'vector store: {vector_store}')
        storage_context = self.get_storage_context(vector_store)
        # self.logger.info(f'storage context: {storage_context}')
        self.logger.info('Index in vector store')
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=self.service_context, embed_model = self.embed_model
        )
        return index

    def feed(self, url):
        self.logger.info(f'Feed {self.name} from url: {url}')
        documents = self.get_documents(url)
        index = self.feed_from_documents(documents)
        self.print_stats()
        return index

    def feed_from_documents(self, documents):
        self.logger.info(f'Feed documents to Raptor: {self.name}')
        index_name = self.index_from_name()
        return self.index_documents(index_name, documents)

    def index_from_name(self):
        return f'{self.name}{INDEX_SUFFIX}'

    def raptor_table(self):
        return f'{INDEX_TABLE_PREFIX}{self.index_from_name()}'

    def get_documents(self, url):
        self.logger.info(f'Getting documents from: {url}')
        RemoteReader = download_loader("RemoteReader")
        loader = RemoteReader()
        return loader.load_data(url=url)

    def query(self, question):
        self.logger.info('query')
        index = self.get_index()
        return self.query_from_index(index, question)

    def query_from_index(self, index, question):
        self.logger.info('query_from_index')
        query_engine = index.as_query_engine(service_context=self.service_context)
        response = query_engine.query(question)
        self.logger.info(f'Reponse: {response.response}')
        self.logger.info(f'Metadata: {response.metadata}')
        self.print_stats()
        return response

    def print_stats(self):
        cost_embeddings = COST_1000_EMBEDDINGS * self.token_counter.total_embedding_token_count / 1000
        cost_prompt = COST_1000_PROMPT * self.token_counter.prompt_llm_token_count / 1000
        cost_completion = COST_1000_COMPLETION * self.token_counter.completion_llm_token_count / 1000
        cost_total = cost_embeddings + cost_prompt + cost_completion
        self.logger.info('STATS')
        self.logger.info('|_ TOKENS')
        self.logger.info('|___ Embedding Tokens       : ' + str(self.token_counter.total_embedding_token_count))
        self.logger.info('|___ LLM Prompt Tokens      : ' + str(self.token_counter.prompt_llm_token_count))
        self.logger.info('|___ LLM Completion Tokens  : ' + str(self.token_counter.completion_llm_token_count))
        self.logger.info('|___ Total LLM Token Count  : ' + str(self.token_counter.total_llm_token_count))
        self.logger.info('|_ COSTS')
        self.logger.info('|___ Embedding              : ' + str(cost_embeddings))
        self.logger.info('|___ LLM Prompt             : ' + str(cost_prompt))
        self.logger.info('|___ LLM Completion         : ' + str(cost_completion))
        self.logger.info('|___ Total                  : ' + str(cost_total))


    def suicide(self):
        self.logger.info('suicide')
        table = self.raptor_table()
        self.logger.info(f'dropping table: {table}')
        with self.db_connect_index.cursor() as c:
            c.execute(f'DROP table IF EXISTS "{table}"')
        return 'arg'


raptorai = RaptorAI()
