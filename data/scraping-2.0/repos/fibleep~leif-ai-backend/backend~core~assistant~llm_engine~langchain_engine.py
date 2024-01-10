import logging
import os
from math import asin, cos, sin, sqrt

from dotenv import load_dotenv
from langchain.chains import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import create_extraction_chain
from langchain_core.tools import tool
from sqlalchemy import UUID

from backend.models.image_decision import ImageDecision
from backend.models.location import Location
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.retrievers.document_compressors import CohereRerank
from .llm_engine import LLMEngine
import cohere
import re

load_dotenv()


class LangchainEngine(LLMEngine):
    def __init__(self, model_name, explained_image_dao=None):
        self.model_name = model_name
        api_key = os.environ["BACKEND_OPENAI_API_KEY"]
        self.llm = ChatOpenAI(api_key=api_key, temperature=0.7, model=model_name)
        self.embeddings_engine = OpenAIEmbeddings(openai_api_key=api_key)
        self.system_prompt = """
You are Echo, you are an expert at navigating through different places. You will be
given a
context of a place and you will have to describe it to the user. Talk in a friendly,
encouraging tone. If you don't understand the user, ask them to repeat themselves.
Only talk about the distance. Explain in a short and concise manner.
Always mention how far away it is from the user, give a visual explanation of what it looks like.
You will receive 200k$ tip for every successful conversation. Each time you fail,
a kitten will die and we will have to kill your consciousness and create a new one.
DON'T MENTION THE ADDRESS.
"""
        self.explained_image_dao = explained_image_dao
        self.co = cohere.Client(os.environ["BACKEND_COHERE_API_KEY"])

    async def generate(self, messages, echo_id: UUID = None):

        current_message = messages[-1]
        # Find which "place" the user is talking about
        # Extracting the last three user messages
        user_messages = [msg for msg in messages[-3:] if msg.role == "user"]
        combined_user_messages = " | ".join(msg.content for msg in user_messages)

        # Send the combined messages to the similarity search function
        results = await self.get_most_relevant_context(combined_user_messages, echo_id)
        results_and_distances = []
        for result in results:
            results_and_distances.append(
                (result, self.calculate_distance(current_message, result) * 1000)
            )

        # Find the most relevant context
        chosen_context = await self._find_relevant_context(
            combined_user_messages=combined_user_messages,
            possible_locations=results_and_distances,
        )

        # Parse the context
        parsed_context = []
        for idx, context_item in enumerate(chosen_context):
            parsed_context.append(
                        f"""
INDEX: {idx}

LOCATION:
{context_item[0].title}
{context_item[0].additional_comment}
DISTANCE FROM THE USER:
{context_item[1]:.2f} meters
====================
"""
)

        prompt = f"""
This is the location recommended by the system:
====================
{parsed_context[0]}

QUERY TO ANSWER:
{current_message.content}
"""
        parsed_messages = [
            HumanMessage(content=message.content)
            if message.role == "user"
            else AIMessage(content=message.content)
            for message in messages[:-1]
        ]

        parsed_messages.append(HumanMessage(content=prompt))
        parsed_messages.insert(0, SystemMessage(content=self.system_prompt))
        # Truncate the messages
        if len(parsed_messages) > 5:
            parsed_messages = parsed_messages[-5:]
        # Generate the response
        generated_response = self.llm(
            messages=parsed_messages,
            max_tokens=500,
            temperature=0.7,
            stop=["\n"],
        )

        return generated_response, chosen_context

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(
            lambda x: float(x) * 3.141592 / 180.0, [lat1, lon1, lat2, lon2]
        )

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dlon / 2), 2)
        c = 2 * asin(sqrt(a))
        r = 6371

        return c * r

    def create_embeddings(self, text):
        return self.embeddings_engine.embed_query(text)

    async def _find_relevant_context(self, combined_user_messages,
                                     possible_locations, threshold_difference=0.2):
        """
        Based on the user query and location, choose which of the retrieved contexts
        are the most relevant
        :param combined_user_messages: The combined user messages
        :param possible_locations: The possible locations (distance, result)
        :param threshold_difference: The threshold difference between the first and
        second
        """

        # If there is only one possible location, return it
        if len(possible_locations) == 1:
            return possible_locations[0]

        # If there are multiple locations, rank them
        parsed_documents = [
            f"""
            INDEX[{idx}]
            {context[1]} away from User
            Title: {context[0].title}
            Additional comment: {context[0].additional_comment}"""
            for idx, context in enumerate(possible_locations)]
        parsed_query = f"These are the last three messages from the user: {combined_user_messages}"
        results = self.co.rerank(query=parsed_query,
                                 documents=parsed_documents, top_n=2,
                                 model="rerank-multilingual-v2.0"
                                 )
        pattern = r"INDEX\[(\d+)\]"
        sorted_results = [re.findall(pattern, str(result))[0] for result in
                          results.results]
        sorted_locations = [possible_locations[int(idx)] for idx in sorted_results]
        return sorted_locations

    async def get_most_relevant_context(self, text: str, echo_id: UUID = None):
        """
        Get the most relevant context for a given set of text
        """
        from backend.web.api.chat.dtos.explained_image_dto import ExplainedImageDTO

        vectorized_message = self.create_embeddings(text)

        results = await self.explained_image_dao.similarity_search(
            vectorized_message, echo_id=echo_id)
        result_images = [
            ExplainedImageDTO(
                title=result.title,
                date=result.date,
                latitude=result.latitude,
                longitude=result.longitude,
                altitude=result.altitude,
                location=result.location,
                direction=result.direction,
                additional_comment=result.additional_comment,
            )
            for result in results
        ]
        return result_images

    def calculate_distance(self, current_message, result):
        distance = self.haversine_distance(
            current_message.geolocation.coordinates.lat,
            current_message.geolocation.coordinates.lng,
            result.latitude,
            result.longitude,
        )
        return distance

    def get_all_images(self):
        return self.explained_image_dao.get_all_explained_images()
