"""
written by:   Eugene M.
              https://github.com/apexDev37

date:         dec-2023

demo:         Holy-Prompt:  precursively optimize user prompts with AI
                            before querying an LLM.
"""

import os

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage

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

    user_prompt = input("[prompt] >>> ")

    # Ensure prompt consists of the following:
    # desired action and precise context.
    template = """
      You're an assistant tasked with transforming standard prompts from users into high-quality inputs 
      tailored for Language Models (LLMs). Utilize the principles of prompt engineering, ensuring your 
      redesigned version demonstrates scientific-based, first principles such as Contextual Relevance, 
      Specificity and Clarity, Demonstration Effect, Length and Complexity, and Creativity. Your goal is to 
      enhance the user's prompt for optimal LLM comprehension and performance. Ensure that the transformed 
      prompt does not misinterpret, but rather retains and expands on the user's intent and desired outcome.

      User Prompt: `{user_prompt}`
      {format_instructions}
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # Leverage parser to parse the LLM response
    # into a Python native type our script can consume.
    ai_prompt_schema = ResponseSchema(
        name="ai_prompt",
        description="The LLM's transformed and enhanced version of the user prompt",
    )

    schemas = [ai_prompt_schema]
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = output_parser.get_format_instructions(only_json=True)

    messages = prompt_template.format_messages(
        user_prompt=user_prompt, format_instructions=format_instructions
    )

    # Initial response that contains
    # the LLM's enhanced version of the user prompt.
    response = chat(messages)

    # Python native type our app can use.
    data = output_parser.parse(response.content)
    print(data)

    # Leverage the LLM's optimized prompt to improve the LLM's
    # output with better input.
    messages = [HumanMessage(content=data["ai_prompt"])]
    response = chat(messages)
    print(response.content)


if __name__ == "__main__":
    main()
