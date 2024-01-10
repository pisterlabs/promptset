"""
Demo of Langchain components: models, prompts, and parsers.

written by:   Eugene M.
              https://github.com/apexDev37

date:         nov-2023

usage:        simple demo of using a prompt templates and output parser.
"""

import os

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

CHATGPT_MODEL: str = "gpt-3.5-turbo"


def main() -> None:
    """Script entry-point func."""

    # Load your Open AI, API key
    # If you don't have one, see:
    # https://platform.openai.com/account/api-keys
    _ = load_dotenv(dotenv_path=".envs/keys.env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize model: Open AI's GPT LLM
    chat = ChatOpenAI(
        temperature=0.0,
        model=CHATGPT_MODEL,
    )

    # Ensure prompt consists of the following:
    # desired action and precise context
    greeting = "Jambo"
    template = """
      Analyze the text enclosed in backticks to identify whether it constitutes a typical and formal greeting.
      If the text is indeed a greeting, provide information on the language it is in, three synonyms for the greeting, and specify five countries where it is commonly used.

      If the text is not a greeting, kindly suggest three universally recognized and formal greetings from around the world at random.
      Text: `{customer_greeting}`
      {format_instructions}
    """

    prompt_template = ChatPromptTemplate.from_template(template)

    # Leverage parser to parse the LLM response
    # into a Python native type our script can consume.
    greeting_schema = ResponseSchema(
        name="greeting",
        description="True, if the text is a typical and formal greeting, else False if not or unknown",
    )

    schemas = [greeting_schema]
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = output_parser.get_format_instructions()

    messages = prompt_template.format_messages(
        customer_greeting=greeting, format_instructions=format_instructions
    )
    response = chat(messages)

    # Python native type our app can use.
    data = output_parser.parse(response.content)
    print(data)


if __name__ == "__main__":
    main()
