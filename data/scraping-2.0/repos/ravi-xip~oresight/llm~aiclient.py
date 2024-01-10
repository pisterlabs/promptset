import json
import logging
import os
from hashlib import md5
from typing import List

import backoff as backoff

import openai
from openai.embeddings_utils import get_embedding

from config.settings import OPENAI_CHAT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, FALLBACK_ANSWER, \
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from llm.conversation.conversation import Conversation
from llm.prompts.prompts import BIO_EXTRACTION_PROMPT_TMPL, REDDIT_PROSPECT_CLASSIFIER_PROMPT_TMPL, \
    REDDIT_PROSPECT_JSON_EXTRACTOR_TMPL, PROSPECT_ANSWER_PROMPT_TMPL, SUMMARY_GENERATOR_PROMPT_TMP
from llm.util import Utils
from reader.file import File


class AIClient:
    def __init__(self):
        self._model = OPENAI_CHAT_MODEL
        self._temperature = DEFAULT_TEMPERATURE
        self._max_tokens = DEFAULT_MAX_TOKENS
        self._system_prompt = SYSTEM_PROMPT
        self._fallback_answer = FALLBACK_ANSWER
        self._hits = 0
        self._thor_client = None
        self._openai_client = openai

    # Create a setter for model
    @property
    def model(self) -> str:
        return self._model
    @model.setter
    def model(self, model: str):
        self._model = model


    def extract_bio(self, doc_contents: str) -> dict:
        """
        Given the contents of a page (i.e. could be derived from a Webpage), extracts entities from the contents
        These entities are extracted in the format of a JSON dictionary.

        :param doc_contents:
        :return:
        """
        prompt = BIO_EXTRACTION_PROMPT_TMPL.format(text=doc_contents)
        response = self.__run(prompt)
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            response_json = {}
        return response_json

    def extract_info(self, doc_contents: str) -> dict:
        """
        Given the contents of a page (i.e. could be derived from a Webpage), extracts entities from the contents
        These entities are extracted in the format of a JSON dictionary.

        :param doc_contents:
        :return:
        """
        logging.debug(f'extract_info: doc_contents: {doc_contents}')
        prompt = REDDIT_PROSPECT_CLASSIFIER_PROMPT_TMPL.format(text=doc_contents)
        response = self.__run(prompt)
        prompt_json_extractor = REDDIT_PROSPECT_JSON_EXTRACTOR_TMPL.format(text=response)
        response_json = self.__run(prompt_json_extractor)
        logging.debug(f'extract_info: response: {response}')
        try:
            response_json = json.loads(response_json)
            logging.debug(f'extract_info: response_json: {response_json}')
        except json.JSONDecodeError:
            logging.error(f'Could not decode response')
            response_json = {}
        return response_json

    def explain(self, query: str, conversation: str, chunks: List[str]) -> str:
        """
        Given a collection of chunks and a query, generates an explanation for the query.
        :param query:
        :param conversation:
        :param chunks:
        :return:
        """
        prompt = PROSPECT_ANSWER_PROMPT_TMPL.format(prospects=chunks, conversation=conversation, question=query)
        response = self.__run(prompt)
        return response

    def summarize(self, doc_contents: str) -> str:
        """
        Given a document, generates a summary for the document.
        :param doc_contents: The document for which the summary needs to be generated.
        :return:
        """
        prompt = SUMMARY_GENERATOR_PROMPT_TMP.format(text=doc_contents)
        response = self.__run(prompt)
        return response

    def embedding(self, document: str) -> list[float]:
        """
        Given a document, generates an embedding vector for the document.
        :param document: The document for which the embedding vector needs to be generated.
        :return: The embedding vector for the document.
        """
        # Normalizing the document by replacing \n by spaces
        document = document.replace('\n', ' ')

        # Step I: Check if the embedding vector is already present in Redis.
        if self._thor_client and self._thor_client.exists(document):
            self._hits += 1
            return self._thor_client.get_array(document)

        # Step II: If not present, then generate the embedding vector and store it in Redis.
        openai.api_key = OPENAI_API_KEY
        query_embedding = get_embedding(document, engine=OPENAI_EMBEDDING_MODEL)

        # Step III: Save the embedding vector in Redis.
        if self._thor_client:
            self._thor_client.set_array(document, query_embedding)

        # Step IV: Return the embedding vector.
        return query_embedding

    def __run(self, prompt: str) -> str:
        """
        This function is used to generate a response from the OpenAI API.
        :param prompt: The command to be executed.
        :return: The response from the OpenAI API.
        """
        logging.debug(f'__run: prompt: {prompt}')
        conversation_string = (f'[{{"role": "system", "content": "{self._system_prompt}"}},'
                               f'{{"role": "user", "content": "{Utils.normalize(prompt)}"}}]')
        return self.__chat(Conversation(conversation_string))

    def __chat(self, conversation: Conversation) -> str:
        """
        Given a conversation, generates a response from the OpenAI API.
        :param conversation: The conversation to be used to generate the response.

        :return: A response for the query asked by the user in the conversation.
        """
        # Step I: Check if the explanation is already present in Redis
        hash_key = md5(conversation.__str__().encode('utf-8')).hexdigest()
        if self._thor_client and self._thor_client.exists(hash_key):
            self._hits += 1
            return self._thor_client.get(hash_key).decode('utf-8')

        # Step II: Generate the response from the OpenAI API
        response = self.__chat_with_backoff(
            model=self._model,
            messages=conversation.get_messages(),
            max_tokens=self._max_tokens
        )

        # Step III: Extract the response from the OpenAI API
        response_text = self.__extract_response(response)

        # Step IV: Store the response in Redis
        if self._thor_client:
            self._thor_client.set(hash_key, response_text)

        # Step V: Return the explanation to the client
        return response_text

    def __extract_response(self, response: openai.ChatCompletion) -> str:
        """
        Extract the response from the OpenAI API.
        :param response: ChatCompletion object from the OpenAI API that contains the response.
        :return:
        """
        response_text = (response.choices[0].message.content if response.choices else '').strip()
        return response_text or self._fallback_answer

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def __chat_with_backoff(self, **kwargs):
        return self._openai_client.ChatCompletion.create(**kwargs)


if __name__ == '__main__':
    ai = AIClient()
    file = File()
    # Find the path to the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, 'data/crawl.txt')
    if not os.path.exists(path):
        raise Exception(f'File {path} does not exist.')
    contents = file.read(path, normalize=True)
    print(ai.extract_info(contents))
