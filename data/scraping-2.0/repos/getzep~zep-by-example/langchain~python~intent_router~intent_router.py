from typing import List, Mapping, NamedTuple, Optional

from langchain import LLMChain
from langchain.chains.router import MultiRouteChain
from langchain.chains.router.base import RouterChain
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch


class IntentModel(NamedTuple):
    """A model for an intent that a human may have."""

    intent: str
    description: str
    prompt: str
    default: bool = False  # is this the default or fallback intent?


class IntentRouterChain(MultiRouteChain):
    """Chain for routing inputs to different chains based on intent."""

    router_chain: RouterChain
    destination_chains: Mapping[str, LLMChain]
    default_chain: LLMChain

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_intent_models(
        cls,
        intent_models: List[IntentModel],
        llm: ChatOpenAI,
        embedding_model: Optional[Embeddings],
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False,
    ) -> "IntentRouterChain":
        """Create a new IntentRouterChain from a list of intent models."""

        names_and_descriptions = [(i.intent, [i.description]) for i in intent_models]

        router_chain = EmbeddingRouterChain.from_names_and_descriptions(
            names_and_descriptions,
            DocArrayInMemorySearch,
            embedding_model,
            routing_keys=["input"],
            verbose=verbose,
        )

        default_chain: Optional[LLMChain] = None
        destination_chains = {}
        for i in intent_models:
            destination_chains[i.intent] = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    template=i.prompt, input_variables=["input", "chat_history"]
                ),
                memory=memory,
            )
            if i.default:
                default_chain = destination_chains[i.intent]

        if not default_chain:
            raise ValueError("No default chain was specified.")

        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=verbose,
        )
