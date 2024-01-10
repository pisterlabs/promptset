from llama_index import ServiceContext, LLMPredictor, LangchainEmbedding, set_global_service_context
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser
from langchain.chat_models import ChatOpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI

from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    TitleExtractor,
    KeywordExtractor,
)


def init_global_service_context():
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")

    # metadata_extractor = MetadataExtractor(
    #     extractors=[
            # TitleExtractor(nodes=5, llm=llm),
            # QuestionsAnsweredExtractor(questions=3, llm=llm),
            # SummaryExtractor(summaries=["prev", "self"], llm=llm),
            # KeywordExtractor(keywords=10, llm=llm),
    #     ],
    # )

    service_context = ServiceContext.from_defaults(llm=llm)

    set_global_service_context(service_context)
