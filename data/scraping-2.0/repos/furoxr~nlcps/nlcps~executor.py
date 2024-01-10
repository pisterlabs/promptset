from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pydantic.v1 import BaseSettings
from qdrant_client.http.api_client import AsyncApis
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, CreateCollection

from nlcps.analysis_chain import AnalysisChain, AnalysisResult
from nlcps.model import initialize
from nlcps.retrieve_chain import RetrieveChain
from nlcps.types import (
    AnalysisExample,
    ContextRuleExample,
    DSLRuleExample,
    DSLSyntaxExample,
    RetrieveExample,
)
from nlcps.util import logger


class NlcpsConfig(BaseSettings):
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"

    entities: List[str]
    system_instruction: str

    collection_name_prefix: str

    dsl_syntax_k: int = 5
    dsl_rules_k: int = 5
    dsl_examples_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class NlcpsExecutor:
    def __init__(
        self,
        qdrant_client: AsyncApis,
        analysis_chain: AnalysisChain,
        retrieve_chain: RetrieveChain,
    ):
        self.qdrant_client = qdrant_client
        self.analysis_chain = analysis_chain
        self.retrieve_chain = retrieve_chain

    async def init_vectorstore(self):
        """Create collections if not exists."""
        collections = [
            RetrieveExample.collection_name,
            DSLRuleExample.collection_name,
            DSLSyntaxExample.collection_name,
            ContextRuleExample.collection_name,
            AnalysisExample.collection_name,
        ]
        for collection_name in collections:
            try:
                await self.qdrant_client.collections_api.get_collection(collection_name)
                logger.info(f"Collection '{collection_name}' already exists")
            except UnexpectedResponse:
                await self.qdrant_client.collections_api.create_collection(
                    collection_name,
                    create_collection=CreateCollection(
                        vectors=VectorParams(size=1536, distance=Distance.COSINE)
                    ),
                )
                logger.info(f"Collection {collection_name} created.")

    async def analysis(self, user_utterance: str) -> AnalysisResult:
        """Analysis user utterance to get entities and whether context needed."""
        return await self.analysis_chain.run(user_utterance)

    async def retrieve(
        self, user_utterance: str, entities: List[str]
    ) -> List[tuple[RetrieveExample, float]]:
        """Retrieve related samples from sample bank."""
        return await self.retrieve_chain.retrieve_few_shot_examples(
            user_utterance, entities
        )

    async def program_synthesis(
        self,
        user_utterance: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate DSL program to fulfill user utterance."""
        analysis_result = await self.analysis_chain.run(user_utterance)
        logger.debug(f"{analysis_result}")
        if analysis_result.need_context and context is None:
            raise ValueError(
                "User utterance requires context but no context is provided."
            )

        return await self.retrieve_chain.run(
            user_utterance, analysis_result.entities, context
        )


def nlcps_executor_factory(config: NlcpsConfig) -> NlcpsExecutor:
    """Initialize NLCPS executor."""

    init_models(config.collection_name_prefix)

    # Initialize qdrant client, llm client and embedding client
    async_qdrant_client: AsyncApis = AsyncApis(host="http://127.0.0.1:6333")
    llm = ChatOpenAI(
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
    )
    embeddings = OpenAIEmbeddings(  # type: ignore
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
    )
    initialize(async_qdrant_client, embeddings)

    # Initialize executor
    analysis_chain = AnalysisChain(
        llm=llm,
        entities=config.entities,
    )
    retrieve_chain = RetrieveChain(
        llm=llm,
        system_instruction=config.system_instruction,
    )
    return NlcpsExecutor(
        qdrant_client=async_qdrant_client,
        analysis_chain=analysis_chain,
        retrieve_chain=retrieve_chain,
    )


def init_models(prefix: str):
    """Initialize collection name and embedding key of models."""
    DSLSyntaxExample.collection_name = f"{prefix}_dsl_syntax"
    DSLSyntaxExample.embedding_key = "code"

    DSLRuleExample.collection_name = f"{prefix}_dsl_rules"
    DSLRuleExample.embedding_key = "rule"

    RetrieveExample.collection_name = f"{prefix}_dsl_examples"
    RetrieveExample.embedding_key = "user_utterance"

    ContextRuleExample.collection_name = f"{prefix}_context_rules"
    ContextRuleExample.embedding_key = "rule"

    AnalysisExample.collection_name = f"{prefix}_analysis_examples"
    AnalysisExample.embedding_key = "utterance"
