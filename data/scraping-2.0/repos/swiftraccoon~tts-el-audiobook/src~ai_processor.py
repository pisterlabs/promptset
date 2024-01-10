"""
AI Processor Module

This module handles the interaction with OpenAI's API.
It processes text in chunks suitable for LLM context limits.
"""

import logging
import openai


class AIProcessor:
    """
    AI Processor for processing text using OpenAI's ChatGPT API.
    """

    def __init__(self, api_key: str, model: str = "gpt-4", chunk_size: int = 5000):
        """
        Initializes the AI Processor.
        :param api_key: The API key for OpenAI.
        :param model: The model to be used for completions (suitable for chat).
        :param chunk_size: The size of text chunks to process.
        """
        openai.api_key = api_key
        self.model = model
        self.chunk_size = chunk_size

    def process_text(self, text: str) -> str:
        """
        Processes the given text in chunks using OpenAI's ChatGPT API.
        :param text: The text to be processed.
        :return: The processed text.
        :raise Exception: If the API request fails.
        """
        # Split the text into manageable chunks
        chunks = [text[i:i + self.chunk_size]
                  for i in range(0, len(text), self.chunk_size)]

        # Container for all the processed chunks to be joined later
        processed_chunks = []

        for chunk in chunks:
            logging.info(f"Processing chunk of size {len(chunk)}")
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an avid reader diligently converting books to readable text so you can later process them for text to speech conversion. Do not omit any words, but you can change the formatting. For example, you can remove page numbers, headers, footers, and other unnecessary text."},
                    {"role": "user", "content": chunk.strip()},
                ],
            )
            # Extract the response and clean it
            message_content = response['choices'][0]['message']['content']
            processed_chunks.append(message_content)
            logging.debug(f"Processed chunk: {message_content}")

        # Join all the processed chunks into the final string
        return ' '.join(processed_chunks)
