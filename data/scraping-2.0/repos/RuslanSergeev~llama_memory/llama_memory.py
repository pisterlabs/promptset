import os
import argparse
from typing import Dict, List, Any

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index import Document
from llama_index.llms import OpenAI
from llama_index.response.schema import (
    Response, 
    StreamingResponse
)
from llama_index.chat_engine.types import (
    StreamingAgentChatResponse
)
from llama_index.schema import NodeWithScore, BaseNode
from llama_index.node_parser.simple import SimpleNodeParser

from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.text_splitter import SentenceSplitter

# Tried metadata extractors:
#    QuestionsAnsweredExtractor <- not used: didn't show improvements
#    TitleExtractor <- Used previously, KW extractor is better
#    EntityExtractor <- not used: too slow
#    KeywordExtractor <- Used, shows the best improvements
from llama_index.node_parser.extractors.metadata_extractors import (
    KeywordExtractor,
    MetadataExtractor,
)


import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")


class Llama_memory:
    def __init__(
        self,
        load_path: str = None,
        model_config: Dict[str, Any] = {"model": "gpt-4", "temperature": 0.7},
    ):
        self._create_node_parser()
        self._create_service_ctx(model_config)
        self._create_index()
        if load_path:
            self._index_load(load_path)
        self._create_chat_engine()


    def _index_load(self, load_path: str):
        # check memory path exists and not empty
        if not os.path.exists(load_path) or not os.listdir(load_path):
            raise ValueError("Storage path does not exist or is empty")
        try: 
            storage_context = StorageContext.from_defaults(
                persist_dir=load_path
            )
            self.index = load_index_from_storage(
                storage_context=storage_context,
                service_context=self.service_context
            )
        except Exception:
            print("Unable to load memory")

    def _create_index(
        self, 
        nodes: List[BaseNode] = []
    ):
        self.index = VectorStoreIndex(
            nodes,
            service_context=self.service_context
        )

    def _create_chat_engine(
        self
    ):
        self.chat_engine = self.index.as_chat_engine()

    def _retrieve_documents(
        self,
        data_path: str
    ) -> List[Document]:
        filename_meta = lambda filename: {"filename": filename.split(".")[0]}
        if os.path.isdir(data_path):
            reader = SimpleDirectoryReader(
                input_dir=data_path,
                filename_as_id=True,
                file_metadata=filename_meta
            )
        elif os.path.isfile(data_path):
            reader = SimpleDirectoryReader(
                input_files=[data_path],
                filename_as_id=True,
                file_metadata=filename_meta
            )
        else:
            raise ValueError("Invalid data path")
        return reader.load_data()

    def _create_node_parser(self):
#        title_extractor = TitleExtractor(nodes=5)
        kw_extractor = KeywordExtractor(
            keywords = 5
        )
        extractors = [
#            title_extractor,
            kw_extractor
        ]
        metadata_extractor = MetadataExtractor(
            extractors = extractors 
        )
        text_splitter = SentenceSplitter(
            separator=" ",
            paragraph_separator="\n\n",
            chunk_size=1024, 
            chunk_overlap=24
        )
        self.node_parser = SimpleNodeParser.from_defaults(
            metadata_extractor = metadata_extractor,
            text_splitter = text_splitter
        )
#        node_parser = SentenceWindowNodeParser.from_defaults(
#            window_size=5,
#            window_metadata_key="window",
#            metadata_extractor=metadata_extractor,
#        )

    def add_directory(self, directory_path: str):
        documents = self._retrieve_documents(directory_path)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)

    def add_file(self, file_path: str):
        documents = self._retrieve_documents(file_path)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)

    def add_text(
        self, 
        text: str,
        doc_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        doc = Document(text=text)
        if doc_id:
            doc.doc_id = doc_id # type: ignore
        nodes = self.node_parser.get_nodes_from_documents([doc])
        # merge metadata: add if new, expand otherwise
        if metadata:
            for node in nodes:
                for k in metadata:
                    if k not in node.metadata:
                        node.metadata[k] = metadata[k]
                    else:
                        node.metadata[k] += f', {metadata[k]}'
        self.index.insert_nodes(nodes)

    def query(self, query: str) -> Response:
        # return only the suggestions, do not ask the chat.
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            streaming=False
        )
        return query_engine.query(query) # type: ignore

    def query_stream(
        self,
        query: str
    ) -> StreamingResponse:
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            streaming=True
        )
        return query_engine.query(query) # type: ignore

    def query_nodes(
        self,
        query: str
    ) -> List[NodeWithScore]:
        retriever = self.index.as_retriever(
            similarity_top_k=5
        )
        return retriever.retrieve(query)

    def save(self, save_path: str):
        if not self.index:
            raise ValueError("Memory is not initialized")
        self.index.storage_context.persist(save_path)

    def chat_stream(self, query: str) -> StreamingAgentChatResponse:
        # init chat engine if not initialized
        return self.chat_engine.stream_chat(query)

    def _create_service_ctx(
        self,
        model_config: Dict[str, Any] = {"model": "gpt-4", "temperature": 0.1},
    ):
        '''
        Create a service context for the index
        - The llm is the language model used to parse documents into nodes
        and to generate responses to queries
        - The node_parser is used to parse documents into nodes
        - The text_splitter is used to split text into sentences and tokens
        - The metadata_extractors are used to extract metadata from nodes
        '''
        llm = OpenAI(**model_config)
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            node_parser=self.node_parser
        )


def config_log(level: str = "debug"):
    openai.log = level


def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama Memory Management"
    )

    # Positional argument for user query
    parser.add_argument(
        "user_query", 
        nargs="*", 
        type=str, 
        help="User query to be processed."
    )

    # Optional argument for specifying a custom persist directory to load from
    parser.add_argument(
        "--load", 
        type=str, 
        default="storage", 
        help="Directory to load the memory from."
    )

    # Optional argument for specifying a custom persist directory to save to
    parser.add_argument(
        "--save", 
        type=str, 
        help="Directory to save the memory to."
    )

    # Optional argument if user wants to chat with agent
    parser.add_argument(
        "--memory", 
        action="store_true", 
        default=False, 
        help="Just use memory instead full chat"
    )

    # Optional argument for adding files
    parser.add_argument(
        "--add_files", 
        nargs='+', 
        help="List of filenames to be loaded."
    )

    # Optional argument for adding directories
    parser.add_argument(
        "--add_dirs", 
        nargs='+', 
        help="List of directories to be loaded."
    )

    return parser.parse_args()

def print_answer_start():
    print("-" * 20)
    print("🤖: \033[38;5;159m", end="")

def print_answer_end():
    print("\033[0m")
    print("-" * 20)


if __name__ == "__main__":
    args = parse_args()
    persist_dir = args.load

    print("creating memory...")
    m = Llama_memory(persist_dir)

    if args.add_files:
        print("adding files...")
        for file in args.add_files:
            m.add_file(file)

    if args.add_dirs:
        print("adding directories...")
        for directory in args.add_dirs:
            m.add_directory(directory)

    if args.save:
        print("saving memory...")
        m.save(args.save)

    user_msg = ' '.join(args.user_query)
    if args.memory:
        print("querying...")
        rsp = m.query_stream(user_msg)
        print_answer_start()
        rsp.print_response_stream()
        print_answer_end()
    else:
        print("initializing the chat ...")
        while user_msg:
            rsp = m.chat_stream(user_msg)
            print_answer_start()
            rsp.print_response_stream()
            print_answer_end()
            user_msg = input("You: ")
