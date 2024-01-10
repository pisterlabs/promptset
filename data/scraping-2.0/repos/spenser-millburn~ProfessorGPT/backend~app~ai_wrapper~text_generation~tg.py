from openai import OpenAI

"""
TextGeneration Class Summary:

The TextGeneration class facilitates text generation using the OpenAI GPT-3.5 model. It includes methods to generate both JSON and text-based responses based on user prompts and system prompts related to a specified topic.

Attributes:
- client: An instance of the OpenAI API used for text generation.
- topic: Represents the specified topic for text generation.

Methods:
- __init__(self, topic: str): Initializes the TextGeneration class with a specified topic for text generation.
- generate_json(self, user_prompt, system_prompt=None): Generates a JSON response based on user prompts and an optional system prompt related to the specified topic.
- generate_text(self, user_prompt, system_prompt=None): Generates a text-based response based on user prompts and an optional system prompt related to the specified topic.

Usage:
1. Instantiate TextGeneration with a specified topic using: tg = TextGeneration("your_topic_here").
2. Utilize the generate_json method to generate JSON responses based on user and system prompts.
3. Utilize the generate_text method to generate text-based responses based on user and system prompts.

"""

class TextGeneration:
    def __init__(self, topic: str):
        self.client = OpenAI()
        self.topic = topic

    def generate_json(self, user_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant that knows a lot about {self.topic} and only responds with JSON"

        return self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}"},
            ]
        )

    def generate_text(self, user_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant that knows a lot about {self.topic}"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}"},
            ]
        )
        return response.choices[0].message.content


