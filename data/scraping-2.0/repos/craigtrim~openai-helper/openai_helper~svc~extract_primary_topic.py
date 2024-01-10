#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Find and Extract a Primary Topic from an Input Sentence """


from typing import Optional
from functools import lru_cache
from baseblock import BaseObject


class ExtractPrimaryTopic(BaseObject):
    """ Find and Extract a Primary Topic from an Input Sentence """

    __invalid_responses = [
        'the input is incomplete',
        'topic extraction is not possible',
        'does not provide enough information',
        'not related to any specific topic',
        'not a valid question',
        'n/a',
        'no topic',
        'please provide a complete input',
    ]

    def __init__(self):
        """ Change Log

        Created:
            16-Mar-2023
            craigtrim@gmail.com
            #   TODO:   this should likely go into a service called openai-usage
                        that consumes openai-helper and provides custom prompts and extractions like this
        """
        BaseObject.__init__(self, __name__)

    @staticmethod
    @lru_cache
    def _extract(input_text: str) -> Optional[str]:
        from openai_helper import chat

        input_prompt = "Extract the primary topic. Only respond with the topic and no other text.  If you can't find a topic, don't print anything."

        result = chat(input_prompt=input_prompt, messages=input_text)
        if not result or not len(result):
            return None

        return result

    def process(self,
                input_text: str) -> Optional[str]:

        result = self._extract(input_text)
        if not result or not len(result):
            return None

        _result = result.lower().strip()

        if _result.startswith('primary topic:'):
            result = result[14:].strip()

        if _result.startswith('the primary topic is'):
            result = result[20:].strip()

        if _result.startswith('the topic is'):
            result = result[12:].strip()

        for invalid_response in self.__invalid_responses:
            if invalid_response in _result:
                return None

        if _result.startswith('none '):
            return None

        result = result.replace("'", '')
        result = result.replace('"', '')

        if result.endswith('.'):
            result = result[:-1].strip()

        return result
