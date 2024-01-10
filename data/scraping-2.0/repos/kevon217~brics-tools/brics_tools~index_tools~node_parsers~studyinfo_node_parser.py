from llama_index import ServiceContext
from llama_index.llms import OpenAI
import tiktoken
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser, HierarchicalNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    EntityExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)
from llama_index.schema import MetadataMode
from transformers import AutoTokenizer


class StudyInfoNodeParser:
    def __init__(self, config, use_metadata_extractor=True):
        self.config = config
        self.init_service_context()
        self.init_text_splitter()
        if use_metadata_extractor:
            self.init_metadata_extractor()
        self.init_node_parser()
        self.parsed_nodes = []

    def init_service_context(self):
        llm_model_name = self.config.service_context.llm.llm_kwargs.model_name
        temperature = self.config.service_context.llm.llm_kwargs.temperature
        llm = OpenAI(temperature=temperature, model=llm_model_name)
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            context_window=self.config.service_context.llm.context_window,  # TODO: reorganize configs to make service context settings not necessarily specific to metadata extraction in configs
        )

    def init_tokentext_splitter(self):
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=self.config.text_splitter.chunk_size,
            chunk_overlap=self.config.text_splitter.chunk_overlap,
            backup_separators=["\n"],
            tokenizer=tiktoken.encoding_for_model(
                self.config.service_context.llm.llm_kwargs.model_name
            ).encode,
        )

    def init_text_splitter(self):
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=self.config.text_splitter.chunk_size,
            chunk_overlap=self.config.text_splitter.chunk_overlap,
            backup_separators=["\n"],
            tokenizer=AutoTokenizer.from_pretrained(
                self.config.service_context.embedding.model_name
            ),
        )

    def init_metadata_extractor(self):
        llm = self.service_context.llm  # Assuming llm is part of the service_context
        self.metadata_extractor = MetadataExtractor(
            extractors=[
                KeywordExtractor(
                    keywords=self.config.metadata_extractor.keywords,
                    llm=llm,
                ),
            ],
        )

    def init_node_parser(self):
        metadata_extractor = getattr(self, "metadata_extractor", None)
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter,
            include_prev_next_rel=self.config.include_prev_next_rel,
            include_metadata=bool(metadata_extractor),
            metadata_extractor=metadata_extractor,
        )

    def parse_nodes_from_documents(self, documents):
        batch_size = self.config.batch_size
        parsed_nodes = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                batch_nodes = self.node_parser.get_nodes_from_documents(
                    batch, show_progress=True
                )  # NOTE: get_nodes_from_documents is deprecated
                parsed_nodes.extend(batch_nodes)
            except Exception as e:
                print(f"Failed to process batch {i} to {i + batch_size}. Error: {e}")
        # for n in parsed_nodes:
        #     n.relationships[DocumentRelationship.SOURCE] = n.get_doc_id()
        self.parsed_nodes = parsed_nodes
        return parsed_nodes
