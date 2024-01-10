import logging
import os
from statistics import mode

import openai
from dotenv import load_dotenv

from supabase_db import supabase
from utils import read_chat_history

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RelevanceAgent:
    """
    This class is responsible for determining the relevance of a given stimulus.
    It uses the OpenAI API to categorize the stimulus into a topic.
    If no existing topic clearly represents the stimulus, it creates a new topic.
    """

    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        openai.api_key = self.openai_api_key

    def batch_stimulus_into_topic(self, stimulus):
        topics = supabase.table("topic_batches").select("*").execute().data
        topics_str = ", ".join([topic["topic"] for topic in topics])
        logger.info(f"Existing Topics: {topics_str}")
        response = None
        try:
            response = (
                openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                        You are a topic identification system. You receive messages that need to be categorized into topics.
                        If no existing topic clearly represents the message content, create a new topic.
                        Respond with only the string of the topic, no quotes, no fluff, nothing other than the topic itself.

                        Existing Topics: {topics_str}


                        EXAMPLES:
                            Existing Topics: board meeting prep, chiefs game on saturday, dinner with mom, coffee with alex at 4
                            USER: Alex: Who is joining for coffee at 4?
                            ASSISTANT: coffee with alex at 4

                            Existing Topics: brunch with grandma, updates to frontend, new hire onboarding
                            USER: Sara: I'm going to be out of the office for the next two weeks
                            ASSISTANT: Sara out of office
                        """,  # noqa: E501
                        },
                        {
                            "role": "user",
                            "content": f"{stimulus.author}: {stimulus.content}",
                        },
                    ],
                )
                .choices[0]
                .message
            )
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")

        # Update topic batches in db
        self.upsert_topic_batches(stimulus, topics, response["content"])

    def upsert_topic_batches(self, stimulus, topics, topic):
        existing = []
        old_relevance = None
        for entry in topics:
            if entry["topic"] == topic:
                existing.append(entry["stimuli"])
                old_relevance = entry["relevance"]
                break
        stimuli = existing + [stimulus]
        new_relevance = self.calculate_relevance(topic, stimuli, old_relevance)
        batch = {
            "topic": topic,
            "stimuli": [s.dict() for s in stimuli],
            "relevance": new_relevance,
        }
        try:
            response = supabase.table("topic_batches").upsert(batch).execute().data
            return response
        except Exception as e:
            logger.error(f"Error upserting topic batch: {e}")

    def calculate_relevance(self, topic, stimuli, old_relevance):
        messages_str = read_chat_history()
        print(stimuli)
        priorities = [stimulus.priority for stimulus in stimuli]
        topic_priority = mode(priorities)
        if "critical" in priorities:
            topic_priority = "critical"
        response = None
        try:
            response = (
                openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                        You are a proficient system that determines how relevant a topic is.
                        You know the recent chat history between a human user and their AI assistant.
                        You receive a Topic, it's Old Relevance score, and a Priority level.
                        Your job is to determine the new Relevance score for the topic in the current context of the given conversation.
                        Choose only one of the following Relevance scores as your output:
                        - Ignorable
                        - Low
                        - Moderate
                        - High
                        - Crucial

                        Follow these steps to determine the new Relevance score:
                        1. Identify the 3 most salient concepts in the chat history.
                        2. Evaluate how similar the Topic is to each of the 3 concepts.
                        3a. If the Old Relevance score is UNNASSIGNED, choose a Relevance score based on the Topic similarity to chat history.
                        3b. Otherwise, choose a Relevance score based on the Topic similarity to chat history and the Priority level.
                        4. If the Priority level is critical, always choose Crucial.

                        Respond with only the string of the new Relevance score, no quotes, no fluff, nothing other than the Relevance itself.

                        Chat History:
                        {messages_str}
                        """,  # noqa: E501
                        },
                        {
                            "role": "user",
                            "content": f"""Topic: {topic}
                        Old Relevance: {old_relevance if old_relevance else "UNNASSIGNED"}
                        Priority: {topic_priority}""",  # noqa: E501
                        },
                    ],
                )
                .choices[0]
                .message
            )
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")

        logger.info(response)
        # Return the relevance score for the topic batch
        return response.content.lower()
