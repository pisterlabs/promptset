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

from typing import Any, Dict, cast

import openai
import requests
from aea.configurations.base import PublicId
from aea.connections.base import BaseSyncConnection
from aea.mail.base import Envelope
from aea.protocols.base import Address, Message
from aea.protocols.dialogue.base import Dialogue

from packages.valory.protocols.llm.dialogues import LlmDialogue
from packages.valory.protocols.llm.dialogues import LlmDialogues as BaseLlmDialogues
from packages.valory.protocols.llm.message import LlmMessage


PUBLIC_ID = PublicId.from_str("valory/openai:0.1.0")

ENGINES = {
    "chat": ["gpt-3.5-turbo", "gpt-4"],
    "completion": ["text-davinci-002", "text-davinci-003"],
}


class LlmDialogues(BaseLlmDialogues):
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
            return LlmDialogue.Role.CONNECTION

        BaseLlmDialogues.__init__(
            self,
            self_address=str(kwargs.pop("connection_id")),
            role_from_first_message=role_from_first_message,
            **kwargs,
        )


class OpenaiConnection(BaseSyncConnection):
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
            for setting in (
                "openai_api_key",
                "engine",
                "max_tokens",
                "temperature",
                "request_timeout",
                "use_staging_api",
                "staging_api"
            )
        }
        openai.api_key = self.openai_settings["openai_api_key"]
        self.dialogues = LlmDialogues(connection_id=PUBLIC_ID)

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
        llm_message = cast(LlmMessage, envelope.message)

        dialogue = self.dialogues.update(llm_message)

        if llm_message.performative != LlmMessage.Performative.REQUEST:
            self.logger.error(
                f"Performative `{llm_message.performative.value}` is not supported."
            )
            return

        try:
            value = self._get_response(
                prompt_template=llm_message.prompt_template,
                prompt_values=llm_message.prompt_values,
            )
        except openai.error.AuthenticationError as e:
            self.logger.error(e)
            value = "OpenAI authentication error"
        except openai.error.APIError as e:
            self.logger.error(e)
            value = "OpenAI server error"
        except openai.error.RateLimitError as e:
            self.logger.error(e)
            value = "OpenAI rate limit error"

        response_message = cast(
            LlmMessage,
            dialogue.reply(
                performative=LlmMessage.Performative.RESPONSE,
                target_message=llm_message,
                value=value,
            ),
        )

        response_envelope = Envelope(
            to=envelope.sender,
            sender=envelope.to,
            message=response_message,
            context=envelope.context,
        )

        self.put_envelope(response_envelope)

    def _get_response(self, prompt_template: str, prompt_values: Dict[str, str]):
        """Get response from openai."""

        # Format the prompt using input variables and prompt_values
        formatted_prompt = prompt_template.format(**prompt_values) if prompt_values else prompt_template
        engine = self.openai_settings["engine"]

        # Call the staging API
        if self.openai_settings["use_staging_api"]:
            url = self.openai_settings["staging_api"]
            response = requests.post(
                url,
                json={"engine": engine, "prompt": formatted_prompt},
                timeout=self.openai_settings["request_timeout"]
            )
            return response.json()["text"]

        # Call the OpenAI API
        if engine in ENGINES["chat"]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt},
            ]
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=self.openai_settings["temperature"],
                max_tokens=self.openai_settings["max_tokens"],
                n=1,
                request_timeout=self.openai_settings["request_timeout"],
                stop=None,
            )
            output = response.choices[0].message.content
        elif engine in ENGINES["completion"]:
            response = openai.Completion.create(
                engine=engine,
                prompt=formatted_prompt,
                temperature=self.openai_settings["temperature"],
                max_tokens=self.openai_settings["max_tokens"],
                n=1,
                request_timeout=self.openai_settings["request_timeout"],
                stop=None,
            )
            output = response.choices[0].text
        else:
            raise AttributeError(f"Unrecognized OpenAI engine: {engine}")

        return output

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
