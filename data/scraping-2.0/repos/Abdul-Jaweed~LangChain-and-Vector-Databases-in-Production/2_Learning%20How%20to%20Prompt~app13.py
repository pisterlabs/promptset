# CommaSeparatedOutputParser

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()





from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

# Prepare the Prompt

template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

model = OpenAI(
    openai_api_key=apikey,
    model_name='text-davinci-003',
    temperature=0.0
)

# Send the Request
output = model(model_input)
parser.parse(output)