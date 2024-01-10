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
# from brics_tools.index_tools.index_managers.studyinfo_summary_manager import (
#     StudyInfoSummaryIndexManager,
# )
from brics_tools.index_tools.index_loaders.studyinfo_index_loader import (
    StudyInfoVectorStoreIndexLoader,
)
from brics_tools.index_tools.query_engines.studyinfo_query_engine import (
    StudyInfoQueryEngine,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


def build_index(df, cfg):
    # Step 1: Load Data
    loader = StudyInfoLoader(cfg.loaders.studyinfo_loader)
    loader.load_studies()
    df_studyinfo = loader.df_studyinfo
    print(df_studyinfo.shape)

    # Step 2: Preprocess

    # Step 3: Create Documents
    doc_creator = StudyInfoDocumentCreator(cfg.document_creators.studyinfo_document)
    docs = doc_creator.create_documents(loader.df_studyinfo)
    print(f"Number of documents: {len(docs)}")

    # Step 4: Parse Documents into Nodes

    # simple docstore nodes
    node_parser = StudyInfoNodeParser(
        cfg.node_parsers.studyinfo_nodes, use_metadata_extractor=False
    )
    nodes = node_parser.parse_nodes_from_documents(docs)
    storage_context = StorageContext.from_defaults(docstore=SimpleDocumentStore())
    storage_context.docstore.add_documents(nodes)
    storage_context.persist(
        persist_dir=cfg.index_managers.studyinfo_vectorstore_index.storage_context.storage_path_root
    )

    # vectorstore index nodes
    node_parser_vectorstore = StudyInfoNodeParser(
        cfg.node_parsers.studyinfo_nodes, use_metadata_extractor=True
    )
    nodes_vectorstore = node_parser_vectorstore.parse_nodes_from_documents(docs)

    # # summary index nodes
    # node_parser_summary = StudyInfoNodeParser(cfg.node_parsers.studyinfo_nodes, use_metadata_extractor=False)
    # nodes_summary = node_parser_summary.parse_nodes_from_documents(docs)

    # Step 5: Create Llama Indices

    # create vectorstore index from nodes
    vectorstore_mngr = StudyInfoVectorStoreIndexManager(
        cfg.index_managers.studyinfo_vectorstore_index, nodes_vectorstore
    )
    vectorstore_index = vectorstore_mngr.create_index()

    # # create summary index from docs
    # summary_mngr = StudyInfoSummaryIndexManager(cfg.index_managers.studyinfo_summary_index, docs)
    # summary_mngr.create_index()
    # summary_mngr.persist_index()


if __name__ == "__main__":
    index = build_index(cfg)
