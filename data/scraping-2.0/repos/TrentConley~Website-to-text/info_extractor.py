import os
import openai
from dotenv import load_dotenv


class InformationExtractor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key")
        openai.api_key = self.api_key

    def extract(self, query, text):
        # Format the messages for the language model
        messages = [
            {
                "role": "system",
                "content": "You are given a certin text that comes from a screenshot of a website. The user is looking for their query. Even if you only capture part of it, return that. Keep your answers brief and just to the information they requested.",
            },
            {"role": "user", "content": f"Text: {text}"},
            {"role": "user", "content": f"Query: {query}"},
        ]

        # Call the OpenAI API
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)

        # Extract the information from the response
        result = response.choices[0].message["content"]
        return result

    def extract_from_history(self, past_extractions, query):
        # Initialize an empty list to store the messages
        messages = [
            {
                "role": "system",
                "content": "You are given a list of past extractions from screenshots of websites. The user is looking for their query in this pool of information. Keep your answers brief and just to the information they requested.",
            },
        ]

        # Add each past extraction to the messages
        for extraction in past_extractions:
            messages.append(
                {"role": "user", "content": f"Past Extraction: {extraction}"}
            )

        # Add the query to the messages
        messages.append({"role": "user", "content": f"Query: {query}"})

        # Call the OpenAI API
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)

        # Extract the information from the response
        result = response.choices[0].message["content"]
        return result
