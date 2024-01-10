import logging
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.pydantic_v1 import Field
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.vectorstores import Qdrant
from langchain.vectorstores.base import VectorStoreRetriever

from memories.memory_prompts import base_example_souvenirs, helper_sys_prompt

load_dotenv()
logger = logging.getLogger(__name__)

helper_llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    streaming=False,
)


def get_rand_time(weeks=10):
    # get random datetime object ranging from x weeks ago to today
    start_date = datetime.now() - timedelta(weeks=weeks)
    end_date = datetime.now()
    return start_date + (end_date - start_date) * random.random()


souvenirs = [
    Document(page_content=souvenir, metadata={"created_at": get_rand_time()}) for souvenir in base_example_souvenirs
]

# Long Term Memory (Vectors)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
qdrant = Qdrant.from_documents(
    documents=souvenirs[:1],
    embedding=embeddings,
    # path="data/LTM/",
    location=":memory:",
    collection_name="LTM",
    force_recreate=False,
)
retriever = TimeWeightedVectorStoreRetriever(vectorstore=qdrant, decay_rate=0.1, k=5)
retriever.add_documents(souvenirs)

# LTM = VectorStoreRetrieverMemory(retriever=retriever, memory_key="LTM")


###########
class CustomMemory(BaseChatMemory):
    """short term memory with token limit + long term memory with vector store retriever."""

    human_prefix: str = "Jean"
    ai_prefix: str = "Cyrano"
    llm: BaseLanguageModel
    memory_key: str = "memory"
    return_messages: bool = True
    verbose: bool = False

    # STM args
    max_token_limit: int = 3000

    # LTM args
    retriever: TimeWeightedVectorStoreRetriever = Field(exclude=True)
    reflection_threshold: Optional[float] = None
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories."""

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is False."""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is True."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def _get_summary_importance(self, memory_content: str):
        helper_sys = SystemMessage(
            content=helper_sys_prompt,
        )

        summary_response = self.llm(
            [
                helper_sys,
                HumanMessage(
                    content=(
                        "Résume le contexte suivant de facon concise, du point de vue de Cyrano."
                        f"\nContexte :\n'''\n{memory_content}\n'''"
                    )
                ),
            ]
        )
        summary = summary_response.content

        importance_response = self.llm(
            [
                helper_sys,
                HumanMessage(
                    content=(
                        "Sur une échelle de 1 à 10, où 1 est purement banal "
                        "(ex: lance un timer, commence une partie d'échec) "
                        "et 10 est extrêmement important (ex: une information personnelle, "
                        "un évenement mondial). Evalue l'importance de l'extrait. "
                        "Répond par un seul nombre entier."
                        f"\nContexte :\n'''\n{summary}\n'''"
                        "\nScore : "
                    )
                ),
            ]
        )

        score = importance_response.content

        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return summary, (float(match.group(1)) / 10) * self.importance_weight
        else:
            return summary, 0.0

    def _retrieve_ltm(self, query) -> None:
        """Update the buffer with the long term memory."""

        now = datetime.now()
        docs = self.retriever.get_relevant_documents(query=query, now=now)

        ltm_prompt = "\n---\nSouvenirs de précédentes conversations, à utiliser dans la réponse si pertinent :\n"

        return ltm_prompt + "\n".join([doc.page_content for doc in docs])

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_str = inputs["input"]
        # self.update_buffer_with_ltm(docs)
        input_str += self._retrieve_ltm(input_str)
        self.chat_memory.add_user_message(input_str)
        return {self.memory_key: self.buffer}

    def _form_documents(self, pruned_memory: List[BaseMessage]) -> List[Document]:
        text = get_buffer_string(
            pruned_memory,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
        summary, importance_score = self._get_summary_importance(text)
        if self.verbose:
            logger.info(f"souvenir of {len(summary)} chrs added")
            logger.info(f"'''{len(summary)}'''")
        return [Document(page_content=summary, meta={"created_at": datetime.now(), "importance": importance_score})]

    def _store_long_term_memory(self, pruned_memory: List[BaseMessage]) -> None:
        """Save context from this conversation to document store."""
        documents = self._form_documents(pruned_memory=pruned_memory)
        self.retriever.add_documents(documents)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        buffer = self.chat_memory.messages
        buffer.pop()  # pop (input + ltm (retrieved docs))
        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self._store_long_term_memory(pruned_memory=pruned_memory)

    # def add_memories(self, memory_content: str, now: Optional[datetime] = None) -> List[str]:
    #     """Add an observations or memories to the agent's memory."""
    #     importance_scores = self._score_memories_importance(memory_content)

    #     self.aggregate_importance += max(importance_scores)
    #     memory_list = memory_content.split(";")
    #     documents = []

    #     for i in range(len(memory_list)):
    #         documents.append(
    #             Document(
    #                 page_content=memory_list[i],
    #                 metadata={"importance": importance_scores[i]},
    #             )
    #         )

    #     result = self.retriever.add_documents(documents, current_time=now)


def init_memory(max_token_limit: int = 1500):
    return CustomMemory(
        llm=helper_llm,
        retriever=retriever,
        max_token_limit=max_token_limit,
        verbose=True,
    )
