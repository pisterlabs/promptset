#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""OpenAI connection and channel."""
import json
import random
import time
from typing import Any, Dict, List, cast

import faiss
import numpy as np
import openai
from aea.configurations.base import PublicId
from aea.connections.base import BaseSyncConnection
from aea.mail.base import Envelope
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue

from packages.algovera.protocols.chat_completion.dialogues import ChatCompletionDialogue
from packages.algovera.protocols.chat_completion.dialogues import (
    ChatCompletionDialogues as BaseChatCompletionDialogues,
)
from packages.algovera.protocols.chat_completion.message import ChatCompletionMessage


PUBLIC_ID = PublicId.from_str("algovera/chat_completion:0.1.0")

### PROMPTS ###
modify_question_prompt = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:
"""

# system_prompt
system_prompt = """
Use the following pieces of context to answer the question at the end.
Additionally, you are also given the previous conversation history to help you answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

# user prompt with no history
user_prompt = """
Question: {question}

Context: {context}

Chat History: {chat_history}
"""
### PROMPTS ###


class ChatCompletionDialogues(BaseChatCompletionDialogues):
    """A class to keep track of IPFS dialogues."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize dialogues.

        :param kwargs: keyword arguments
        """

        def role_from_first_message(  # pylint: disable=unused-argument
            message: Message, receiver_address: Address
        ) -> Dialogue.Role:
            """Infer the role of the agent from an incoming/outgoing first message

            :param message: an incoming/outgoing first message
            :param receiver_address: the address of the receiving agent
            :return: The role of the agent
            """
            return ChatCompletionDialogue.Role.CONNECTION

        BaseChatCompletionDialogues.__init__(
            self,
            self_address=str(kwargs.pop("connection_id")),
            role_from_first_message=role_from_first_message,
            **kwargs,
        )


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (
        openai.error.APIError,
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
        openai.error.ServiceUnavailableError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded. Last exception: {e}"
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print(f"Exception: {e}")
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    """Chat completion with exponential backoff."""
    return openai.ChatCompletion.create(**kwargs)


@retry_with_exponential_backoff
def embeddings_with_backoff(**kwargs):
    """Embedding with exponential backoff."""
    return openai.Embedding.create(
        **kwargs,
    )


class ChatCompletionConnection(BaseSyncConnection):
    """Proxy to the functionality of the openai SDK."""

    MAX_WORKER_THREADS = 1
    connection_id = PUBLIC_ID

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """
        Initialize the connection.

        The configuration must be specified if and only if the following
        parameters are None: connection_id, excluded_protocols or restricted_to_protocols.

        Possible arguments:
        - configuration: the connection configuration.
        - data_dir: directory where to put local files.
        - identity: the identity object held by the agent.
        - crypto_store: the crypto store for encrypted communication.
        - restricted_to_protocols: the set of protocols ids of the only supported protocols for this connection.
        - excluded_protocols: the set of protocols ids that we want to exclude for this connection.

        :param args: arguments passed to component base
        :param kwargs: keyword arguments passed to component base
        """
        super().__init__(*args, **kwargs)
        self.openai_settings = {
            setting: self.configuration.config.get(setting)
            for setting in ("openai_api_key", "model", "max_tokens", "temperature")
        }
        openai.api_key = self.openai_settings["openai_api_key"]
        self.dialogues = ChatCompletionDialogues(connection_id=PUBLIC_ID)

    def main(self) -> None:
        """
        Run synchronous code in background.

        SyncConnection `main()` usage:
        The idea of the `main` method in the sync connection
        is to provide for a way to actively generate messages by the connection via the `put_envelope` method.

        A simple example is the generation of a message every second:
        ```
        while self.is_connected:
            envelope = make_envelope_for_current_time()
            self.put_enevelope(envelope)
            time.sleep(1)
        ```
        In this case, the connection will generate a message every second
        regardless of envelopes sent to the connection by the agent.
        For instance, this way one can implement periodically polling some internet resources
        and generate envelopes for the agent if some updates are available.
        Another example is the case where there is some framework that runs blocking
        code and provides a callback on some internal event.
        This blocking code can be executed in the main function and new envelops
        can be created in the event callback.
        """

    def on_send(self, envelope: Envelope) -> None:
        """
        Send an envelope.

        :param envelope: the envelope to send.
        """
        chat_completion_message = cast(ChatCompletionMessage, envelope.message)

        dialogue = self.dialogues.update(chat_completion_message)

        if (
            chat_completion_message.performative
            != ChatCompletionMessage.Performative.REQUEST
        ):
            self.logger.error(
                f"Performative `{chat_completion_message.performative.value}` is not supported."
            )
            return

        self.logger.info("Processing LLM request...")

        # Get response from OpenAI API
        response = self.get_response(
            request_message=json.loads(chat_completion_message.request["request"]),
        )

        # Send response
        response_message = cast(
            ChatCompletionMessage,
            dialogue.reply(
                performative=ChatCompletionMessage.Performative.RESPONSE,
                target_message=chat_completion_message,
                response=response,
            ),
        )

        response_envelope = Envelope(
            to=envelope.sender,
            sender=envelope.to,
            message=response_message,
            context=envelope.context,
        )

        self.put_envelope(response_envelope)

    def get_response(self, request_message: Dict) -> Dict:
        """
        Get request message and return response based on request type.
        args:
            request_message: request message
        """
        try:
            request_type = request_message.get("request_type")
            assert request_type is not None, "request_type is not specified."

            if not request_type:
                raise Exception("request_type is not specified.")

            if request_type == "chat_completion":
                self.logger.info("Processing chat completion request...")
                return self.cc_repsonse(request_message)

            elif request_type == "embedding":
                self.logger.info("Processing embedding request...")
                return self.embedding_response(request_message)

            elif request_type == "chat":
                self.logger.info("Processing chat request...")
                return self.chat_response(request_message)

            else:
                raise Exception(f"request_type `{request_type}` is not supported.")

        except Exception as e:
            reponse = {
                "error": "True",
                "error_name": str(e.__class__.__name__),
                "error_message": str(e),
            }
            return reponse

    def embedding_response(self, embedding_request: Dict) -> Dict:
        """
        Make an embedding from a embedding request.
        args:
            embedding_request: embedding request
        """

        # Get chunks
        chunks = embedding_request.get("chunks")
        assert chunks is not None, "chunks is not specified."

        # Shoud this be something user can specify?
        EMBEDDING_MODEL = "text-embedding-ada-002"
        BATCH_SIZE = 1000

        # Make chunks to embeddings mapping
        chunk_to_embedding = {}
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = chunks[batch_start:batch_end]
            print(f"Batch {batch_start} to {batch_end-1}")
            response = embeddings_with_backoff(model=EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]
            batch_embeddings = [e["embedding"] for e in response["data"]]
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk_to_embedding[chunk] = embedding

        # Make response package
        response_package = {
            "chunks_to_embeddings": json.dumps(chunk_to_embedding),
            "error": "False",
        }

        return response_package

    def cc_repsonse(self, cc_request: Dict) -> Dict:
        """
        Get chat_completion response from OpenAI.
        Does not use any context and memory.
        Just a simple chat completion.
        args:
            cc_request: cc request
        """

        # Get messages
        system_message = cc_request.get("system_message")
        user_message = cc_request.get("user_message")

        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Get response from OpenAI API
        raw_response = completions_with_backoff(
            model=self.openai_settings["model"],
            messages=messages,
            temperature=self.openai_settings["temperature"],
        )

        response = raw_response.choices[0]["message"]["content"]

        # Make response package
        response_package = {
            "response": response,
            "error": "False",
        }
        self.logger.info("Chat completion response: {}".format(response_package))
        return response_package

    def chat_response(self, chat_request: Dict) -> Dict:
        """
        Get chat response from OpenAI.
        Includes context and memory.
        If chat_history exists, makes a modified query first.
        args:
            chat_request: chat request
        """

        # Get chat history, query, c2e
        chat_history = chat_request.get("chat_history", [])
        query = chat_request.get("question")
        chunks_to_embeddings = chat_request.get("chunks_to_embeddings")
        self.logger.info(f"Chat history: {chat_history}\n\nQuery: {query}")

        # Modify query if chat history is not empty
        if chat_history:
            query = self.modify_query_response(
                modify_query_request={
                    "chat_history": self.chat_history_to_string(chat_history),
                    "question": query,
                }
            )
            self.logger.info(f"Modified query: {query}")

        # Use similarity search to find similar chunks
        r_docs = self.find_similar_chunks(
            query=query,
            chunk_to_embedding=chunks_to_embeddings,
        )

        # Make messages for chat
        messages = self._chat_response(
            query=query,
            chat_history=chat_history,
            retrieved_docs=r_docs,
        )

        # Get response from OpenAI using the messages
        raw_response = completions_with_backoff(
            model=self.openai_settings["model"],
            messages=messages,
            temperature=self.openai_settings["temperature"],
        )

        response = raw_response["choices"][0]["message"]["content"]

        # Append new query and response to chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Return response
        response_package = {
            "response": response,
            "chat_history": chat_history,
            "modified_query": query,
            "error": "False",
        }
        self.logger.info("Chat response: {}".format(response_package))

        return response_package

    def concatenate_contexts(self, contexts: List) -> str:
        """Function to concatenate contexts"""
        return "\n\n".join(contexts)

    def chat_history_to_string(self, chat_history: List) -> str:
        """Function to convert chat history to a string"""
        chat_history_string = ""
        for message in chat_history:
            chat_history_string += f"{message['role']}: {message['content']}\n"
        return chat_history_string

    def _chat_response(
        self, query: str, chat_history: List, retrieved_docs: List
    ) -> List:
        """Function to generate messages for chat"""
        context = self.concatenate_contexts(retrieved_docs)
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if chat_history:
            ch_string = self.chat_history_to_string(chat_history)
        else:
            ch_string = ""

        messages.append(
            {
                "role": "user",
                "content": user_prompt.format(
                    question=query,
                    context=context,
                    chat_history=ch_string,
                ),
            }
        )

        return messages

    def modify_query_response(self, modify_query_request: Dict) -> str:
        """
        Get modify_query response from OpenAI.
        Used to modify a question to be a standalone question based on previous conversation.
        args:
            modify_query_request: modify_query request
        """
        chat_history = modify_query_request.get("chat_history", [])
        assert chat_history is not None, "chat_history is not specified."

        question = modify_query_request.get("question")

        prompt = modify_question_prompt.format(
            chat_history=chat_history,
            question=question,
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        raw_response = completions_with_backoff(
            model=self.openai_settings["model"],
            messages=messages,
            temperature=self.openai_settings["temperature"],
        )
        response = raw_response["choices"][0]["message"]["content"]

        return response

    def find_similar_chunks(
        self, query: str, chunk_to_embedding: Dict, k: int = 4
    ) -> List:
        """Similarity search to find similar chunks to a query"""

        EMBEDDING_MODEL = "text-embedding-ada-002"
        BATCH_SIZE = 1000

        query_embedding = embeddings_with_backoff(model=EMBEDDING_MODEL, input=query,)[
            "data"
        ][0]["embedding"]

        index = faiss.IndexFlatIP(1536)
        index.add(np.array(list(chunk_to_embedding.values())))
        D, I = index.search(np.array([query_embedding]), k)

        return [list(chunk_to_embedding.keys())[i] for i in I[0]]

    def on_connect(self) -> None:
        """
        Tear down the connection.

        Connection status set automatically.
        """

    def on_disconnect(self) -> None:
        """
        Tear down the connection.

        Connection status set automatically.
        """
