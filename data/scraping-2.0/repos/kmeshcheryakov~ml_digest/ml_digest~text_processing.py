"""
text_processing.py

This module provides functions for text processing tasks, such as summarizing articles and
extracting topics from questions. It leverages the GPT-3.5-turbo model from OpenAI to perform
these tasks.
"""

from typing import Dict

import openai


async def summarize_article(article: Dict[str, str]) -> Dict[str, str]:
    """
    Summarize an article and identify its main topic.

    :param article: A dictionary containing article information.
    :return: A dictionary with the summarized article and its main topic.
    """
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize the following article concisely but with enough information for creating a digest later."
            "Also, identify the main topics of the article. Please, put the topics comma separated on a new line, like in the following example:"
            "The article reports a breakthrough in neuroscience, as an international team has released the complete brain connectivity map of a young fruit fly,"
            "which contains 3,016 neurons and 548,000 synapses, making it the most complex whole-brain wiring diagram to date. This map, also known as a connectome, could help reveal secrets for developing better AI. \n"
            "Topics: neuroscience, brain map, insect, fruit fly, connectome, synapses, complex wiring diagram, AI.",
        },
        {
            "role": "user",
            "content": f"Title: {article['title']}.\nDescription: {' '.join(article['description'].split(' ')[:1500])}.",
        },
    ]

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=conversation,
    )

    response_text = response["choices"][0]["message"]["content"]
    try:
        summary, topic = response_text.split("\n", 1)
    except ValueError:
        summary = response_text
        topic = ""
    return {"summary": summary, "topic": topic.strip()}


async def extract_topic_from_question(question: str) -> str:
    """
    Identify the main topic of a question.

    :param question: A string containing the question.
    :return: A string containing the main topic of the question.
    """
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Identify the main topic of this question:",
        },
        {"role": "user", "content": question},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
    )

    topic = response["choices"][0]["message"]["content"]
    return topic.strip()
