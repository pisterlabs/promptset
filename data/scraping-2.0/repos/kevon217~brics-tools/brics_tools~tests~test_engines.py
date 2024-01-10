import os
from dotenv import load_dotenv
import openai

from llama_index import (
    download_loader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler
from llama_index.callbacks.open_inference_callback import as_dataframe, QueryData
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore

from brics_tools.utils import helper
from brics_tools.index_tools.loaders.studyinfo_loader import StudyInfoLoader
from brics_tools.index_tools.document_creators.studyinfo_document_creator import (
    StudyInfoDocumentCreator,
)
from brics_tools.index_tools.node_parsers.studyinfo_node_parser import (
    StudyInfoNodeParser,
)
from brics_tools.index_tools.index_managers.studyinfo_vectorstore_manager import (
    StudyInfoVectorStoreIndexManager,
)
from brics_tools.index_tools.index_managers.studyinfo_summary_manager import (
    StudyInfoSummaryIndexManager,
)
from brics_tools.index_tools.index_loaders.studyinfo_index_loader import (
    StudyInfoVectorStoreIndexLoader,
)
from brics_tools.index_tools.query_engines.studyinfo_query_engine import (
    StudyInfoQueryEngine,
)
from llama_index.indices.query.schema import QueryBundle


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


# Initialize Query Engine

# engine_default = StudyInfoQueryEngine.from_defaults(config=cfg)
engine = StudyInfoQueryEngine(cfg)
engine.init_vector_index()

rtrv = engine.create_retriever_only_engine(similarity_top_k=20, rerank_top_n=5)
rtrv = engine.retriever_engine


rag = engine.create_retriever_query_engine(
    similarity_top_k=20, rerank_top_n=10, top_n_for_llm=7
)
rag = engine.query_engine

cbh = engine.vector_index_manager.callback_handler


# TESTS
# r1 = rtrv.query("risk factors associated with PTE")
# r1 = rtrv.retrieve(QueryBundle("TRACK-TBI"))
r1 = rtrv.query(QueryBundle("TRACK-TBI"))
# r1 = rtrv.query("TBD")
# r1 = rtrv.query("informatics")
# r1 = rtrv.query("Depolarization Inhibition")

print(f"# of source_nodes: {len(r1.source_nodes)}")
r1.source_nodes
rtrv_scores = []
for n in r1.source_nodes:
    title = n.metadata["title"]
    rtrv_scores.append((title, n.score))
rtrv_scores


# rag1 = rag.query("TRACK-TBI")
# rag1 = rag.query("Find me studies that involve deploraization inhibition")
rag1 = rag.query("tell me about the track-tbi studies")

print(f"# of source_nodes: {len(rag1.source_nodes)}")
rag1.source_nodes
rag_scores = []
for n in rag1.source_nodes:
    title = n.metadata["title"]
    rag_scores.append((title, n.score))
rag_scores


query_data_buffer = cbh.flush_query_data_buffer()
query_dataframe = as_dataframe(query_data_buffer)
query_dataframe
node_data_buffer = cbh.flush_node_data_buffer()
node_dataframe = as_dataframe(node_data_buffer)
node_dataframe
