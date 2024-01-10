#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Run a Chat Completion against OpenAI """


from typing import Any
from typing import List
from typing import Optional

from pprint import pformat

from baseblock import Enforcer
from baseblock import Stopwatch
from baseblock import BaseObject

from openai.error import RateLimitError
from openai.error import PermissionError
from openai.error import AuthenticationError
from openai.error import ServiceUnavailableError

from openai_helper.dmo import ChatMessageFormatter


class RunChatCompletion(BaseObject):
    """ Run a Chat Completion against OpenAI """

    def __init__(self,
                 conn: object):
        """ Change Log

        Created:
            1-Mar-2023
            craigtrim@gmail.com
            *   https://github.com/craigtrim/openai-helper/issues/9
        Updated:
            28-Mar-2023
            craigtrim@gmail.com
            *   pass model name in dynamically

        Args:
            conn (object): a connected instance of OpenAI
            timeout (int, optional): the timeout for the API call. Defaults to 15.
        """
        BaseObject.__init__(self, __name__)
        self._completion = conn.ChatCompletion.create
        self._formatter = ChatMessageFormatter().process

    def _process(self,
                 input_messages: List[str],
                 model: str) -> Optional[dict]:

        def invoke_call() -> Optional[Any]:
            try:

                return self._completion(
                    model=model,
                    messages=input_messages
                )

                # DESIGN NOTE
                # Do not catch Error, Exception, or general error classes
                # force this on the consumer ...

            except RateLimitError as e:
                self.logger.exception('Rate Limit Error', e)
                return None

            except PermissionError as e:
                self.logger.exception('Permission Error', e)
                return None

            except AuthenticationError as e:
                self.logger.exception('Authentication Error', e)
                return None

            except ServiceUnavailableError as e:
                self.logger.exception('Service Unavailable Error', e)
                return None

        response = invoke_call()

        if not response:
            return {
                'input': input_messages,
                'output': None
            }

        return {
            'input': input_messages,
            'output': dict(response)
        }

    def process(self,
                input_prompt: str,
                messages: List[str],
                model: Optional[str] = 'gpt-3.5-turbo') -> dict:
        """ Run an OpenAI event

        Args:
            input_prompt (str): a defined input prompt

                Sample Input Prompt:
                    "You are a helpful assistant."

            messages (List[str]): The messages to execute the chat completion upon

                Sample Messages:
                    [
                        "Who won the world series in 2020?",
                        "The Los Angeles Dodgers won the World Series in 2020.",
                        "Where was it played?"
                    ]

                There should be an odd-number of messages in the list, with
                    odd-numbered entries as user questions
                    even-numbered entries as system responses

            model (str): the model to use

        Returns:
            dict: an output dictionary with two keys:
                input: the input dictionary with validated parameters and default values where appropriate
                output: the output event from OpenAI
                    Unless RateLimitError, PermissionError, AuthenticationError, ServiceUnavailableError
                    -   each of these errors is gracefully handled, and a dictionary result is still returned with 'output:None'
                    This service will not catch Exception or Error classes generally
                    -   the door is still left open for these and other error types to be thrown
                        and the consumer must plan for this eventuality
        """

        sw = Stopwatch()

        input_messages = self._formatter(
            input_prompt=input_prompt,
            messages=messages)

        d_result = self._process(
            model=model,
            input_messages=input_messages)

        if not d_result:
            self.logger.error('\n'.join([
                'OpenAI Event Execution Failed',
                f'\tTotal Time: {str(sw)}',
                f'\tInput Prompt: {input_prompt}',
                f'\tMessages:\n{pformat(messages)}']))

        if self.isEnabledForDebug:
            Enforcer.is_dict(d_result)
            self.logger.debug('\n'.join([
                'OpenAI Event Execution Completed',
                f'\tTotal Time: {str(sw)}',
                f'\tInput Prompt: {input_prompt}',
                f'\tMessages:\n{pformat(messages)}',
                f'\tOutput Result:\n{pformat(d_result)}']))

        return d_result
