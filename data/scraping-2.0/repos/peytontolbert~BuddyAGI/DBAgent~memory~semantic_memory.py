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
            Apple Computer was founded on April 1, 1976, by Steve Wozniak, Steve Jobs and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. 
            It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. 
            The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. 
            Apple Computer went public in 1980 to instant financial success. 
            The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement. 
            By 1985, the high cost of its products, and power struggles between executives, caused problems.
            Wozniak stepped back from Apple Computer amicably and pursued other ventures, while Jobs resigned bitterly and founded NeXT, taking some Apple Computer employees with him.
            RESPONSE:
            {{
                "Apple Computer Company": "a company founded in 1976 by Steve Wozniak, Steve Jobs, and Ronald Wayne to develop and sell personal computers",
                "Steve Wozniak": "an American inventor, electronics engineer, programmer, philanthropist, and technology entrepreneur who co-founded Apple Computer Company with Steve Jobs",
                "Steve Jobs": "an American entrepreneur, business magnate, inventor, and industrial designer who co-founded Apple Computer Company with Steve Wozniak and Ronald Wayne, and later founded NeXT",
                "Ronald Wayne": "an American retired electronics industry worker and co-founder of Apple Computer Company, who left the company after only 12 days"
            }}
            [INPUT TEXT]
            """


        SCHEMA_TEMPLATE = """
            You are an AI assistant reading a input text and trying to extract entities from it.
            Extract ONLY proper nouns from the input text and return them as a JSON object.
            You should definitely extract all names and places.
            [RULE]
            Your response must be provided exclusively in the JSON format outlined below, without any exceptions. 
            Any additional text, explanations, or apologies outside of the JSON structure will not be accepted. 
            Please ensure the response adheres to the specified format and can be successfully parsed by Python's json.loads function.

            Strictly adhere to this JSON RESPONSE FORMAT for your response.
            Failure to comply with this format will result in an invalid response. 
            Please ensure your output strictly follows RESPONSE FORMAT.

            [JSON RESPONSE FORMAT]
            schema = {{
                "entity1": "description of entity1. Please describe the entities using sentences rather than single words.",
                "entity2": "description of entity2. Please describe the entities using sentences rather than single words.",
                "entity3": "description of entity3. Please describe the entities using sentences rather than single words."
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
