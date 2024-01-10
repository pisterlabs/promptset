from dive.langchain.agents.agent import Agent
from langchain.vectorstores import Chroma
from dive.util.chromaDBClient import ChromaDBClient
from langchain.chains.summarize import load_summarize_chain
from dive.util.configAPIKey import set_openai_api_key
from langchain import OpenAI
from dive.indices.service_context import ServiceContext
from dive.indices.index_context import IndexContext
from dive.retrievers.query_context import QueryContext
from dive.types import EmbeddingModel
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import importlib


class VectorStoreRetrieverAgent(Agent):

    def __init__(self):
        set_openai_api_key()

    def retrieve(self, prompt) -> str:
        similarity_search = "similarity"
        refine_chain_type = "refine"
        llm = OpenAI()
        self.index_test_data(chunk_size=256,chunk_overlap=20,embedding_function=OpenAIEmbeddings())
        query_text = "What did the author do growing up?"
        service_context = ServiceContext.from_defaults(embeddings=OpenAIEmbeddings(),llm=llm)
        query_context = QueryContext.from_defaults(service_context=service_context)
        relevant_docs = query_context.query(query=query_text,k=4,filter={'connector_id': "example"})
        #I wrote a basic summarization function, which does not take external llm yet, it should and it should call langchain
        summary=query_context.summarization(documents=relevant_docs)
        chain = load_summarize_chain(llm = llm, chain_type= refine_chain_type, prompt = prompt)
        return chain.run()



    def index_test_data(self,chunk_size, chunk_overlap, embeddings):
        package_name = "integrations.connectors.example.filestorage.request_data"
        mod = importlib.import_module(package_name)
        data = mod.load_objects(None, None, "paul_graham_essay", None, None, None)

        metadata = {'account_id': 'self', 'connector_id': 'example',
                    'obj_type': 'paul_graham_essay'}
        _ids = []
        _documents = []
        for d in data['results']:
            _metadata = metadata
            if 'metadata' in d:
                _metadata.update(d['metadata'])
            document = Document(page_content=str(d['data']), metadata=_metadata)
            _documents.append(document)
            _ids.append(d['id'])

        embedding_model = EmbeddingModel()
        embedding_model.chunking_type = "custom"
        embedding_model.chunk_size = chunk_size
        embedding_model.chunk_overlap = chunk_overlap
        service_context = ServiceContext.from_defaults(embed_config=embedding_model,embeddings=embeddings)
        IndexContext.from_documents(documents=_documents, ids=_ids, service_context=service_context)






