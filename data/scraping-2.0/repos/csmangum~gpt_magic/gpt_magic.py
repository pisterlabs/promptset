import os

import openai
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

openai.api_key = os.environ["OPENAI_API_KEY"]


@magics_class
class GPT_Magic(Magics):

    @line_magic
    def gpt_magic(self, line):
        response = openai.Completion.create(
            engine='gpt-4',
            prompt=line)
        print(response.choices[0].text.strip())

    @cell_magic
    def cadabra(self, line, cell):
        return line, cell
