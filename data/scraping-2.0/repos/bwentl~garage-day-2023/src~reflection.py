import sys
import re
from typing import Any, List, Optional, Type, Dict

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts.prompt import PromptTemplate

sys.path.append("./")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler

# from src.tools import ToolHandler
from src.util import get_secrets, get_word_match_list, agent_logs
from src.memory_store import PGMemoryStoreSetter, PGMemoryStoreRetriever
from src.docs import AggregateRetrieval
from src.prompts.memory import REFLECTION_PROMPT, IDENTITY_GENERATION_PROMPT


class Reflection:
    """The Reflection class (1) generate reflections on recent memories and (2) revises self-identity statements"""

    def __init__(self):
        pass

    def generate_reflection(self, memory_store_setter, memory_store_retriever, llm):
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 10,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 2,
            "update_access_time": False,
        }

        recent_memories = AggregateRetrieval(
            vectorstore_retriever=memory_store_retriever
        ).run("", **memory_kwargs)

        reflection_text = "I" + llm(
            REFLECTION_PROMPT.replace("{memory}", recent_memories)
        )

        memory_store_setter.add_memory(
            text=reflection_text,
            llm=pipeline,
            with_importance=True,
            type="reflection",
        )

        return reflection_text

    def generate_new_identity_statement(
        self, memory_store_setter, memory_store_retriever, llm
    ):
        # get previous identity statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity",
        }
        prev_identity_statement_query = memory_store_retriever.get_relevant_documents(
            "", **memory_kwargs
        )
        if len(prev_identity_statement_query) > 0:
            prev_identity_statement = prev_identity_statement_query[0].page_content
        else:
            # initial identity statement
            prev_identity_statement = "This is Llama at Home, Llama for short. I am a generative agent build on an open-source, self-hosted large language model (LLM). I can understand and communicate fluently in English and other languages. I can also provide information, generate content, and help with various tasks. Some of my recent actions include answering questions and telling jokes. My short term plan is to continue chatting with others and learning about the world using tools. My long term plan is to improve my skills and knowledge by learning with tools like web searches and from feedback from others."

        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 5,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 2,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=memory_store_retriever
        ).run("", **memory_kwargs)

        new_identity_statement = llm(
            IDENTITY_GENERATION_PROMPT.replace(
                "{reflections}", reflection_input
            ).replace("{prev_identity}", prev_identity_statement)
        )

        memory_store_setter.add_memory(
            text=new_identity_statement,
            llm=pipeline,
            with_importance=True,
            type="identity",
            retrieval_eligible="False",
        )

        return new_identity_statement


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"
    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # Generate reflection
    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)
    reflection = Reflection()
    new_reflection = reflection.generate_reflection(
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
        llm=pipeline,
    )
    print(new_reflection)

    # new_memory_store = PGMemoryStoreRetriever(embedding=eb)
    # memory_kwargs = {
    #     "mem_to_search": 30,
    #     "mem_to_return": 5,
    #     "relevance_wt": 0,
    #     "importance_wt": 0.5,
    #     "recency_wt": 2,
    #     "mem_type": "reflection",
    # }
    # reflection_list = AggregateRetrieval(
    #     vectorstore_retriever=memory_store_retriever
    # ).run("", **memory_kwargs)
    # print(reflection_list)

    new_identity = reflection.generate_new_identity_statement(
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
        llm=pipeline,
    )
    print(new_identity)

    print("done")
