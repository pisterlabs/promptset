import asyncio
import os
import traceback

import json
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from llm.extract_entity.prompt import get_chat_template
from llm.extract_entity.schema import JsonSchema as ENTITY_EXTRACTION_SCHEMA
from llm.json_output_parser import LLMJsonOutputParser, LLMJsonOutputParserException
from logger.hivemind_logger import logger
from ui.cui import CommandlineUserInterface
from utils.constants import DEFAULT_EMBEDDINGS
from utils.util import atimeit, timeit

import base58
CREATE_JSON_SCHEMA_STR = json.dumps(ENTITY_EXTRACTION_SCHEMA.schema)


class SemanticMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    llm: BaseLLM = Field(..., description="llm class for the agent")
    openaichat: Optional[ChatOpenAI] = Field(
        None, description="ChatOpenAI class for the agent"
    )
    embeddings: OpenAIEmbeddings = Field(DEFAULT_EMBEDDINGS,
                                              title="Embeddings to use for tool retrieval",
                                              )
    vector_store: FAISS = Field(
        None, title="Vector store to use for tool retrieval"
    )
    ui: CommandlineUserInterface | None = Field(None)

    class Config:
        arbitrary_types_allowed = True

    # def __init__(self, question: str, **kwargs):
    #     super().__init__(**kwargs)
        # filename = base58.b58encode(question.encode()).decode()
        # if self.vector_store is None:
            # self.vector_store = DeepLake(read_only=True, dataset_path=os.path.join(SEMANTIC_MEMORY_DIR, f"{filename}"),
            #                              embedding=self.embeddings)

    def __del__(self):
        del self.embeddings
        del self.vector_store

    @atimeit
    async def extract_entity(self, text: str, question: str, task: str) -> dict:
        """Extract an entity from a text using the LLM"""
        if self.openaichat:
            # print(f"semantic->extract_entity->Text1: {text}")
            # If OpenAI Chat is available, it is used for higher accuracy results.
            prompt = (
                get_chat_template()
                .format_prompt(text=text, question=question, task=task)
                .to_messages()
            )

            full_prompt = " ".join([msg.content for msg in prompt])

            logger.debug(f"semantic->extract_entity->Prompt: {full_prompt}")
            llm_result = await self.openaichat._agenerate(messages=prompt)
            await self.ui.call_callback_info_llm_result(llm_result)
            result = llm_result.generations[0].message.content
            # result = self.openaichat(prompt).content
        else:
            raise Exception("Should never happen!")

        # Parse and validate the result
        try:
            # print(f"semantic->extract_entity->Result: {result}")
            result_json_obj = LLMJsonOutputParser.parse_and_validate(
                json_str=result, json_schema=CREATE_JSON_SCHEMA_STR, llm=self.llm
            )
        except LLMJsonOutputParserException as e:
            raise LLMJsonOutputParserException(str(e))

        try:
            if len(result_json_obj) > 0:
                await asyncio.create_task(self._embed_knowledge(result_json_obj))

        except BaseException as e:
            print(f"semantic->extract_entity->Text: {text}\n")
            print(f"semantic->extract_entity->Result: {result}\n")
            print(
                f"semantic->extract_entity->Extracted entity: {result_json_obj}\n"
            )
            print(traceback.print_exc())
            # raise Exception(f"Error: {e}")
        return result_json_obj

    @timeit
    def remember_related_knowledge(self, query: str, k: int = 5) -> dict:
        """Remember relevant knowledge for a query."""
        if self.vector_store is None:
            return {}
        relevant_documents = self.vector_store.similarity_search(query, k=k)
        return {
            d.metadata["entity"]: d.metadata["description"] for d in relevant_documents
        }

    @atimeit
    async def _embed_knowledge(self, entity: dict[str:Any]):
        """Embed the knowledge into the vector store."""
        description_list = []
        metadata_list = []

        for entity, description in entity.items():
            description_list.append(description)
            metadata_list.append({"entity": entity, "description": description})

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts=description_list,metadatas=metadata_list,
                                                 embedding=self.embeddings)
            # self.vector_store = DeepLake(read_only=False, dataset_path=SEMANTIC_MEMORY_DIR,
            #                              embedding=self.embeddings)

        self.vector_store.add_texts(texts=description_list, metadatas=metadata_list)

    # async def save_local(self, path: str) -> None:
    #     """Save the vector store to a local folder."""

        # async def _save():
        #     self.vector_store.save_local(folder_path=path)

        # await asyncio.create_task(_save())

    # def load_local(self, path: str) -> None:
    #     """Load the vector store from a local folder."""
    #
    #     # async def _load():
    #     #     self.vector_store = FAISS.load_local(
    #     #         folder_path=path, embeddings=self.embeddings
    #     #     )
    #
    #     # await asyncio.create_task(_load())
    #     self.vector_store = DeepLake(read_only=True, dataset_path=path, embedding=self.embeddings)
