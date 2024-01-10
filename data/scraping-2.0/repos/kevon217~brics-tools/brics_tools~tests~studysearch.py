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
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


# Step 1: Load Data
studyinfo_loader = StudyInfoLoader(cfg.loaders.studyinfo_loader)
studyinfo_loader.load_studies()
df_studyinfo = studyinfo_loader.df_studyinfo

# Step 2: Preprocess


# Step 3: Create Documents
studyinfo_doc_creator = StudyInfoDocumentCreator(
    cfg.document_creators.studyinfo_document
)
studyinfo_docs = studyinfo_doc_creator.create_documents(studyinfo_loader.df_studyinfo)


# Step 4: Parse Documents into Nodes

# simple docstore nodes
studyinfo_node_parser = StudyInfoNodeParser(
    cfg.node_parsers.studyinfo_nodes, use_metadata_extractor=False
)
studyinfo_nodes = studyinfo_node_parser.parse_nodes_from_documents(studyinfo_docs)
storage_context = StorageContext.from_defaults(docstore=SimpleDocumentStore())
storage_context.docstore.add_documents(studyinfo_nodes)
storage_context.persist(
    persist_dir=cfg.index_managers.studyinfo_vectorstore_index.storage_context.storage_path_root
)

# vectorstore index nodes
studyinfo_node_parser_vectorstore = StudyInfoNodeParser(
    cfg.node_parsers.studyinfo_nodes, use_metadata_extractor=True
)
studyinfo_nodes_vectorstore = (
    studyinfo_node_parser_vectorstore.parse_nodes_from_documents(studyinfo_docs)
)

# # summary index nodes
#         0       p -                                                  =sers.studyinfo_nodes, use_metadata_extractor=False)
# studyinfo_nodes_summary = studyinfo_node_parser_summary.parse_nodes_from_documents(studyinfo_docs)

# Step 5: Create Llama Indices

# create vectorstore index from nodes
studyinfo_vectorstore_mngr = StudyInfoVectorStoreIndexManager(
    cfg.index_managers.studyinfo_vectorstore_index, studyinfo_nodes_vectorstore
)
studyinfo_vectorstore_index = studyinfo_vectorstore_mngr.create_index()

# create summary index from docs
studyinfo_summary_mngr = StudyInfoSummaryIndexManager(
    cfg.index_managers.studyinfo_summary_index, studyinfo_docs
)
studyinfo_summary_mngr.create_index()
studyinfo_summary_mngr.persist_index()

# Step 6: Initialize Query Engine
# engine_default = StudyInfoQueryEngine.from_defaults(config=cfg)
engine = StudyInfoQueryEngine(cfg)
# engine.init_summary_index()
engine.init_vector_index()
# engine.init_node_postprocessors()

rtrvr = engine.create_retriever_only_engine(
    similarity_top_k=10, rerank_top_n=None, limit_top_n=5
)
rag = engine.create_retriever_query_engine(
    similarity_top_k=10, rerank_top_n=None, limit_top_n=5
)
rtrvr = engine.retriever_engine
rag = engine.query_engine

# engine.create_query_engine(similarity_top_k=100, response_mode='no_text')
# r1 = engine.query_engine.query("What is the TRACK-TBI study about?")

# engine.init_document_summary_retriever(similarity_top_k=4)
# engine.init_vector_retriever(similarity_top_k=4)
# engine.init_response_synthesizer()
# engine.init_vector_query_engine()
# engine.init_summary_query_engine()

cbh = engine.vector_index_manager.callback_handler


# TESTS
r1 = rtrvr.query("risk factors associated with PTE")
r1 = rtrvr.retrieve(QueryBundle("TRACK-TBI"))
r1 = rtrvr.query("TBD")
r1 = rtrvr.query("informatics")
scores = []
for n in r1.source_nodes:
    title = n.metadata["title"]
    scores.append((title, n.score))

rag1 = rag.query("TRACK-TBI")
scores = []
for n in rag1.source_nodes:
    title = n.metadata["title"]
    scores.append((title, n.score))
print(scores)

sum_retr = engine.retrievers["DocumentSummaryIndexEmbeddingRetriever"]
sr1 = sum_retr.retrieve("What is the TRACK-TBI study about?")

vec_retr = engine.retrievers["VectorIndexRetriever"]
vr2 = vec_retr.retrieve("LE-TBI")


vec_eng = engine.query_engines["vector_query_engine"]
v1 = vec_eng.query("What is TRACK-TBI studying?")

v2 = vec_eng.query("Are there any studies with neuropathology data?")

query_data_buffer = callback_handler.flush_query_data_buffer()
query_dataframe = as_dataframe(query_data_buffer)
query_dataframe
node_data_buffer = callback_handler.flush_node_data_buffer()
node_dataframe = as_dataframe(node_data_buffer)
node_dataframe


sum_eng = engine.query_engines["summary_query_engine"]
s1 = sum_eng.query("TRACK-TBI")


engine_custom.create_query_engine(top_k=5, response_mode="compact")


# Step 7: Query

test_default_1 = engine_default.query(
    "Tell me which studies involve veteran populations."
)
test_default_1.source_nodes
test_custom_1 = engine_custom.query(
    "Tell me which studies involve veteran populations."
)


####################

# RELOAD INDEX


test = studyinfo_docs[0:2]
studyinfo_summary_creator = StudyInfoSummaryIndexManager(
    cfg.index_managers.studyinfo_summary_index, test
)
studyinfo_summary_creator.create_index()
test_index = studyinfo_summary_creator.index

query_engine = studyinfo_summary_creator.index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
response = query_engine.query("I'm looking for studies on exercise.")
response.source_nodes
response.response


engine = StudyInfoQueryEngine(cfg)
engine.init_summary_index()
engine.init_vector_index()
engine.init_document_summary_retriever(similarity_top_k=5)
engine.init_vector_retriever(similarity_top_k=10)
engine.init_response_synthesizer()
engine.init_node_postprocessors()
engine.init_vector_query_engine()
engine.init_summary_query_engine()

# engine.init_retrievers()
# engine.init_query_engines()
vec_eng = engine.query_engines["vector_query_engine"]
v1 = vec_eng.query("What is the LE-TBI study about?")
sum_eng = engine.query_engines["summary_query_engine"]
s1 = sum_eng.query("What is the LE-TBI study about?")

llm_response = v1.response  # need to get better prompting so it doesn't mix up studies.
nodes_result = [node.to_dict() for node in v1.source_nodes]

# sum_eng = engine.summary_query_engine
# vec_eng = engine.vector_query_engine
# q1 = sum_eng.query("What is the TRACK-TBI study about?")
# q2 = vec_eng.query("What is the TRACK-TBI study about?")

sum_retr = engine.retrievers["DocumentSummaryIndexEmbeddingRetriever"]
vec_retr = engine.retrievers["VectorIndexRetriever"]
sr1 = sum_retr.retrieve("What is the LE-TBI study about?")
vr2 = vec_retr.retrieve("What is the LE-TBI study about?")


# For retriever results
test = vr2[0].to_dict()
nodes_result = [node.to_dict() for node in vr2]

# specify metadata to retrieve
# group same studyid results

df = pd.DataFrame(nodes_data)
self.retrieved_nodes = df.sort_values("score", ascending=False)

# engine.init_query_engine_tools()
# engine.init_router_query_engine()


engine = StudyInfoQueryEngine(cfg)
engine.init_summary_index()
engine.init_vector_index()
resp1 = engine.router_query_engine.query("What is the TRACK-TBI study?")
resp2 = engine.router_query_engine.query("What is the COBRIT study about?")
resp3 = engine.router_query_engine.query("Find me studies with severe TBI populations.")
resp4 = engine.router_query_engine.query("Find me all the studies on veterans with TBI")


nodes = test_custom_1.source_nodes
# 1 node/doc
n0 = nodes[0].node
n0_dict = n0.dict()
n0_json = n0.json()
# 3 nodes/doc
n2 = nodes[2].node
n2_dict = n2.dict()
n2_json = n2.json()
n3 = nodes[3].node
n3_dict = n3.dict()
n3_json = n3.json()
n4 = nodes[4].node
n4_dict = n4.dict()
n4_json = n4.json()


keys = [
    "id_",
    "embedding",
    "metadata",
    "excluded_embed_metadata_keys",
    "excluded_llm_metadata_keys",
    "relationships",
    "hash",
    "text",
    "start_char_idx",
    "end_char_idx",
    "text_template",
    "metadata_template",
    "metadata_seperator",
]

study_prefixedId = n4_dict["metadata"]["prefixedId"]
study_title = n4_dict["metadata"]["title"]


engine_default.display_results(test_default_1)
engine_custom.display_results(test_custom_1)


# RELOAD INDEX
studyinfo_vectorstore_mngr = StudyInfoVectorStoreIndexManager(
    cfg.index_managers.studyinfo_vectorstore_index
)
studyinfo_vectorstore_mngr.load_vectorstore_index()
vector_index = studyinfo_vectorstore_mngr.index
vec_qe = vector_index.as_query_engine()
response = vec_qe.query("TRACK-TBI")


studyinfo_summary_mngr = StudyInfoSummaryIndexManager(
    cfg.index_managers.studyinfo_summary_index
)
studyinfo_summary_mngr.load_summary_index()
summary_index = studyinfo_summary_mngr.index
sum_qe = summary_index.as_query_engine()
response = sum_qe.query("TRACK-TBI")
