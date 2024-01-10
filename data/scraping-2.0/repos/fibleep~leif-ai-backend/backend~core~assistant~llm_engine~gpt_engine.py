import logging
import os
from math import asin, cos, sin, sqrt

from dotenv import load_dotenv
from langchain.chains import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import create_extraction_chain
from sqlalchemy import UUID

from backend.models.image_decision import ImageDecision
from backend.models.location import Location
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI

from .llm_engine import LLMEngine

load_dotenv()


class GPTEngine(LLMEngine):
    def __init__(self, model_name, explained_image_dao=None):
        self.model_name = model_name
        api_key = os.environ["BACKEND_OPENAI_API_KEY"]
        self.llm = ChatOpenAI(api_key=api_key, temperature=0.7, model=model_name)
        self.embeddings_engine = OpenAIEmbeddings(openai_api_key=api_key)
        self.system_prompt = """
You are Echo, you will describe different places to the user. You will be given a
context of a place and you will have to describe it to the user. If you don't understand the user, ask them to repeat themselves.
Only talk about the distance. Determine from the context which location is the one the user is talking about.
Always mention how far away it is from the user. DON'T MENTION THE ADDRESS
"""
        self.explained_image_dao = explained_image_dao

    async def generate(self, messages, echo_id: UUID = None):


        current_message = messages[-1]
        logging.info(f"Generating response for message: {current_message}")

        # Find which "place" the user is talking about
        # Extracting the last three user messages
        user_messages = [msg for msg in messages[-3:] if msg.role == "user"]
        combined_user_messages = " ".join(msg.content for msg in user_messages)

        # Send the combined messages to the similarity search function
        results = await self.get_most_relevant_context(combined_user_messages, echo_id)
        results_and_distances = []
        for result in results:
            results_and_distances.append(
                (result, self.calculate_distance(current_message, result) * 1000)
            )

        parsed_context = []
        for result, distance in results_and_distances:
            parsed_context.append(
                f"""
LOCATION:
{result.title}
{result.additional_comment}
DISTANCE FROM THE USER:
{distance:.2f} meters
====================
"""
            )

        prompt = f"""
You are Echo, you will describe different places to the user. You will be given a
context of a place and you will have to describe it to the user. Talk in a friendly,
encouraging tone. If you don't understand the user, ask them to repeat themselves.
Only talk about the distance. Determine from the context which location is the one the user is talking about.
Always mention how far away it is from the user, give a visual explanation of what it looks like.
DON'T MENTION THE ADDRESS.

If you find multiple contexts, rank them and recommend the one that the user is most likely talking about.

====================
{parsed_context}m

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
        [print(message.content) for message in parsed_messages]
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

        return generated_response, results

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
