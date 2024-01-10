from openai import OpenAI
from pymongo.collection import Collection

from utils import generate_embedding

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an expert plasma physicist. You help synthesize the provided context to answer the user questions in a helpful manner. If you don't know the answer, say you don't know.",
}


def query_results(
    openai_client: OpenAI, collection: Collection, field: str, index: str, messages: list[dict[str,str]]
):
    query = messages[-1]["content"]
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": index,
                    "path": field,
                    "queryVector": generate_embedding(
                        openai_client=openai_client, text=query
                    ),
                    "numCandidates": 50,
                    "limit": 1,
                }
            }
        ]
    )

    results = list(results)
    user_message_with_context = "\n# User query:\n" + query
    if len(results) > 0:
        result = results[0]["text"]
        user_message_with_context = (
            "# Context for the user query below:\n" + result + user_message_with_context
        )

    response = openai_client.chat.completions.create(
        model="notNeeded",
        messages=[SYSTEM_PROMPT] + messages[:-1] + [
            {"role": "user", "content": user_message_with_context},
        ],
        n=1,
    )

    return response.choices[0].message.content
