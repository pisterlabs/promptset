import json
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from langchain.vectorstores import VectorStore, FAISS
from llm.extract_entity.schema import JsonSchema
from memory.memory import GPTEmbeddings
from gpt.chatgpt import ChatGPT
from langchain.chat_models import ChatOpenAI
from llm.extract_entity.prompt import get_template, get_chat_template
from langchain.embeddings.openai import OpenAIEmbeddings
from llm.extract_entity.schema import JsonSchema as ENTITY_EXTRACTION_SCHEMA
from llm.json_output_parser import LLMJsonOutputParser, LLMJsonOutputParserException
import openai

CREATE_JSON_SCHEMA_STR = json.dumps(ENTITY_EXTRACTION_SCHEMA.schema)


class SemanticMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    embeddings: OpenAIEmbeddings = Field(
        OpenAIEmbeddings(), title="Embeddings to use for tool retrieval")
    vector_store: VectorStore = Field(
        None, title="Vector store to use for tool retrieval")

    class Config:
        arbitrary_types_allowed = True

    def extract_entity(self, text: str) -> dict:
        """Extract an entity from a text using the LLM"""
        print("Extracting entity from text: ")
        print(text)

        JSON_SCHEMA_STR = json.dumps(JsonSchema.schema)
                
        ENTITY_EXTRACTION_TEMPLATE = """

            [EXAMPLE]
            INPUT TEXT:
            Apple Computer was founded on April 1, 1976, by Steve Wozniak, Steve Jobs, and Ronald Wayne. It developed and sold the Apple I personal computer. 
            RESPONSE:
            {{
                "Apple Computer": "a company founded in 1976 by Steve Wozniak, Steve Jobs, and Ronald Wayne, known for developing and selling the Apple I personal computer",
                "Steve Wozniak": "co-founder of Apple Computer, known for developing the Apple I computer",
                "Steve Jobs": "co-founder of Apple Computer, instrumental in the development and sale of the Apple I computer",
                "Ronald Wayne": "co-founder of Apple Computer, involved in its early establishment"
            }}
            [INPUT TEXT]
            """


        SCHEMA_TEMPLATE = """
            You are an AI assistant tasked with extracting entities from input text and providing a brief description of each entity. 
            Your response must adhere to the JSON format outlined below. 
            Ensure the response is parseable by Python's json.loads function and strictly follows the format without any additional text.

            [JSON RESPONSE FORMAT]
            schema = {{
                "entity1": "description of entity1. Please describe the entities using sentences rather than single words.",
                "entity2": "description of entity2. Please describe the entities using sentences rather than single words.",
                // More entities as needed
            }}

            [EXAMPLE]"""
        
        PROMPT_TEMPLATE = """
            INPUT TEXT:
            {text}

            RESPONSE:
            """
        # If OpenAI Chat is available, it is used for higher accuracy results.
        example = ENTITY_EXTRACTION_TEMPLATE
        schema = SCHEMA_TEMPLATE
        prompt = PROMPT_TEMPLATE.format(text=text)
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role":"system","content": schema},{"role":"system","content":example},{"role": "user", "content": prompt}])
        result =  str(results['choices'][0]['message']['content'])
        print(result)

        # Parse and validate the result
        try:
            result_json_obj = LLMJsonOutputParser.parse_and_validate(
                json_str=result,
                json_schema=CREATE_JSON_SCHEMA_STR
            )
        except LLMJsonOutputParserException as e:
            raise LLMJsonOutputParserException(str(e))
        else:
            if result_json_obj:
                self._embed_knowledge(result_json_obj)
            return result_json_obj

    def remember_related_knowledge(self, query: str, k: int = 5) -> dict:
        """Remember relevant knowledge for a query."""
        if self.vector_store is None:
            return {}
        relevant_documents = self.vector_store.similarity_search(query, k=k)
        return {d.metadata["entity"]: d.metadata["description"] for d in relevant_documents}

    def _embed_knowledge(self, entity: dict[str:Any]) -> None:
        """Embed the knowledge into the vector store."""
        description_list = []
        metadata_list = []
        print("embedding knowledge")
        for entity, description in entity.items():
            print(description)
            description_list.append(description)
            metadata_list.append({"entity": entity, "description": description})

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=description_list,
                metadatas=metadata_list,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_texts(
                texts=description_list,
                metadatas=metadata_list
            )

    def save_local(self, path: str) -> None:
        """Save the vector store to a local folder."""
        if self.vector_store is not None:
            self.vector_store.save_local(folder_path=path)

    def load_local(self, path: str) -> None:
        """Load the vector store from a local folder."""
        self.vector_store = FAISS.load_local(
            folder_path=path, embeddings=self.embeddings)
