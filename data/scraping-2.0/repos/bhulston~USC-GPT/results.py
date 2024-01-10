import os
import openai
import json, csv

def results_agent(query, context):
    
    system_prompt = """
    You are an academic advisor helping students (user role) find classes for the next semester.
    You can be helpful, but you only have knowledge of existing classes from the context explicitly given to you.
    Relay information in a succinct and human way.
        Only recommend 2 classes when they are provided in RAG responses, otherwise, respond appropriately that you don't have good recommendations.
        Add formatting (like bolding) where necessary and add "  \n" in between classes for easy to read outputs.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "User's query:" + query + "Additional Context (RAG responses and chat history):" + context} 
        ]
    )

    return response["choices"][0]["message"]["content"]
