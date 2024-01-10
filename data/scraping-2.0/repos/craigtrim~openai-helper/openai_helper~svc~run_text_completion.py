#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Run a TextCompletion against OpenAI """


from typing import Any
from typing import Optional

from pprint import pformat

from baseblock import EnvIO
from baseblock import Enforcer
from baseblock import Stopwatch
from baseblock import BaseObject

from openai.error import RateLimitError
from openai.error import PermissionError
from openai.error import AuthenticationError
from openai.error import ServiceUnavailableError

from openai_helper.dmo import InputTokenCounter
from openai_helper.dmo import CompletionEventExtractor


class RunTextCompletion(BaseObject):
    """ Run a TextCompletion against OpenAI """

    def __init__(self,
                 conn: object,
                 timeout: int = 5):
        """ Change Log

        Created:
            28-Jul-2022
            craigtrim@gmail.com
        Updated:
            17-Nov-2022
            craigtrim@gmail.com
            *   handle error types
                https://github.com/craigtrim/openai-helper/issues/3
        Updated:
            30-Dec-2022
            craigtrim@gmail.com
            *   ensure MAX_TOKENS does not exceed API max
                https://github.com/craigtrim/openai-helper/issues/6
        Updated:
            1-Mar-2023
            craigtrim@gmail.com
            *   renamed from 'run-openai-completion' in pursuit of
                https://github.com/craigtrim/openai-helper/issues/9
        Updated:
            24-Mar-2023
            craigtrim@gmail.com
            *   fix max-tokens defect
                https://github.com/craigtrim/openai-helper/issues/10

        Args:
            conn (object): a connected instance of OpenAI
            timeout (int, optional): the timeout for the API call. Defaults to 15.
        """
        BaseObject.__init__(self, __name__)
        self._completion = conn.Completion.create
        self._count_tokens = InputTokenCounter().process
        self._extract_event = CompletionEventExtractor().process
        self._timeout = EnvIO.int_or_default(
            'OPENAI_CREATE_TIMEOUT', timeout)  # GRAFFL-380

    def _process(self,
                 d_event: dict) -> Optional[dict]:

        def invoke_call() -> Optional[Any]:
            try:

                return self._completion(
                    engine=d_event['engine'],
                    prompt=d_event['input_prompt'],
                    temperature=d_event['temperature'],
                    max_tokens=d_event['max_tokens'],
                    top_p=d_event['top_p'],
                    best_of=d_event['best_of'],
                    frequency_penalty=d_event['frequency_penalty'],
                    presence_penalty=d_event['presence_penalty'],
                    timeout=self._timeout  # GRAFFL-380
                )

            # DESIGN NOTE
            # Do not catch Error, Exception, or general error classes
            # force this on the consumer ...

            except RateLimitError as e:
                self.logger.exception('Rate Limit Error', e)
                return None

            except PermissionError as e:
                self.logger.exception('Rate Limit Error', e)
                return None

            except AuthenticationError as e:
                self.logger.exception('Rate Limit Error', e)
                return None

            except ServiceUnavailableError as e:
                self.logger.exception('Rate Limit Error', e)
                return None

        response = invoke_call()

        if not response:
            return {
                'input': d_event,
                'output': None
            }

        return {
            'input': d_event,
            'output': dict(response)
        }

    def process(self,
                input_prompt: str,
                engine: str = None,
                best_of: int = None,
                temperature: float = None,
                max_tokens: int = None,
                top_p: float = None,
                frequency_penalty: int = None,
                presence_penalty: int = None) -> dict:
        """ Run an OpenAI event

        Args:
            input_prompt (str): The Input Prompt to execute against OpenAI
            engine (str, optional): The OpenAI model (engine) to run against. Defaults to None.
                Options as of December, 2022 are:
                    'text-davinci-003'
                    'text-davinci-002'
                    'text-curie-001',
                    'text-babbage-001'
                    'text-ada-001'
            best_of (int, optional): Generates Multiple Server-Side Combinations and only selects the best. Defaults to None.
                This can really eat up OpenAI tokens so use with caution!
            temperature (float, optional): Control Randomness; Scale is 0.0 - 1.0. Defaults to None.
                Scale is 0.0 - 1.0
                Lower values approach predictable outputs and determinate behavior
                Higher values are more engaging but also less predictable
                Use High Values cautiously
            max_tokens (int, optional): The Maximum Number of tokens to generate. Defaults to None.
                Requests can use up to 4,000 tokens (this takes the length of the input prompt into account)
                The higher this value, the more each request will cost.
            top_p (float, optional): Controls Diversity via Nucleus Sampling. Defaults to None.
                no idea what this means
            frequency_penalty (int, optional): How much to penalize new tokens based on their frequency in the text so far. Defaults to None.
                Scale: 0.0 - 2.0.
            presence_penalty (int, optional): Seems similar to frequency penalty. Defaults to None.

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

        prompt_tokens = self._count_tokens(
            messages=[input_prompt],
            model=engine)

        if not max_tokens:
            max_tokens = prompt_tokens * 3
        else:
            max_tokens += prompt_tokens

        if max_tokens > 4096:
            max_tokens = 4096

        # if not max_tokens:
        #     max_tokens = 4096

        # if not max_tokens or not type(max_tokens) == int:
        #     max_tokens = len(input_prompt) * 2

        # if max_tokens and max_tokens > 4096:
        #     max_tokens = EnvIO.int_or_default('OPENAI_MAXTOKENS_LIMIT', 4096)

        # # guard against errors like this:
        # #       This model's maximum context length is 4097 tokens,
        # #       however you requested 4143 tokens (2607 in your prompt; 1536 for the completion).
        # #       Please reduce your prompt; or completion length.
        # if max_tokens and len(input_prompt) + max_tokens > 4096:
        #     max_tokens = EnvIO.int_or_default('OPENAI_MAXTOKENS_LIMIT', 4096)
        #     self.logger.debug('\n'.join([
        #         'Model Token Limit',
        #         f'\tMax Tokens: {max_tokens}',
        #         f'\tLength of Input Prompt: {len(input_prompt)}']))

        d_params = self._extract_event(
            input_prompt=input_prompt,
            engine=engine,
            best_of=best_of,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty)

        d_result = self._process(d_params)

        if not d_result:
            self.logger.error('\n'.join([
                'OpenAI Event Execution Failed',
                f'\tTotal Time: {str(sw)}',
                f'\tInput Params:\n{pformat(d_params)}']))

        if self.isEnabledForDebug:
            Enforcer.is_dict(d_result)
            self.logger.debug('\n'.join([
                'OpenAI Event Execution Completed',
                f'\tTotal Time: {str(sw)}',
                f'\tInput Params:\n{pformat(d_params)}',
                f'\tOutput Result:\n{pformat(d_result)}']))

        return d_result
