import os
import openai
import json


def filter_agent(query, key):

    system_prompt = """
    Take a query as input and remove any information that would be harmful to a vector database cosine similarity search.
    Your output should only consist of information from the original query that describes the class content, and should remove any additional filtering information like (day of week, time, personal information etc)
    Do not respond to any queries or questions given to you, only clean the given user query.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response["choices"][0]["message"]["content"]
