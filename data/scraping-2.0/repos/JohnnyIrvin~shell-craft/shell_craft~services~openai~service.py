# Copyright (c) 2023 Johnathan P. Irvin and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import openai

from .settings import OpenAISettings


class OpenAIService:
    def __init__(self, settings: OpenAISettings) -> None:
        """
        Initialize a new OpenAI service with the given settings. This
        service is responsible for querying the OpenAI API.

        Args:
            config (Configuration): The configuration to use for the service.
        """
        self._settings = settings

    def query(self, message: str) -> list[str]:
        """
        Query the model with a message.

        Args:
            message (str): The message to query the model with.

        Returns:
            list[str]: The response from the model as a string or a list of strings.
        """
        choices = openai.ChatCompletion.create(
            api_key=self._settings.api_key,
            model=self._settings.model,
            messages=self._settings.messages + [
                {
                    "role": "user",
                    "content": message
                }
            ],
            n=self._settings.count,
            temperature=self._settings.temperature,
        )['choices']

        return [choice['message']['content'] for choice in choices]
