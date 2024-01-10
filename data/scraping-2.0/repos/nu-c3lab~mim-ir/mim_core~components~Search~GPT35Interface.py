'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''

import os
import openai
from typing import Dict

class GPT35Interface:
    def __init__(self):
        # Init access to OpenAI's API
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

    def generate(self,
                 prompt: str) -> str:
        # Feed the prompt to the language model
        response = self.get_response(prompt)

        # Extract the generation from the response
        qdmr = self.extract_generation_from_response(response)

        return qdmr

    def get_response(self,
                     prompt: str) -> Dict:
        # Make the API call off to GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )

        return response

    def extract_generation_from_response(self,
                                   response: Dict) -> str:
        # Pull out the generated text from the response
        message_content = response["choices"][0]["message"]["content"]

        # Clean up the string
        message_content = message_content.strip()

        return message_content
