"""
written by:   Eugene M.
              https://github.com/apexDev37

date:         dec-2023

usage:        helper utils to support functionality for `demos`.
"""

import os

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils.constants import Commands

MODEL: str = "gpt"


def setup_gpt_model(version: str, variant: str) -> ChatOpenAI:
    """
    Load API key from env file and return specified GPT, Open AI model.
    """

    # If you don't have one, see:
    # https://platform.openai.com/account/api-keys
    _ = load_dotenv(dotenv_path=".envs/keys.env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    engine = "%s-%s-%s" % (MODEL, version, variant)
    return ChatOpenAI(
        temperature=0.7,
        model=engine,
    )


def init_interactive_conversation() -> None:
    """
    Initiate an interactive conversation with LLM and yield each user prompt.
    """

    while True:
        user_input = input("[prompt] >>> ")
        if user_input.strip().lower() in Commands.TERMINATE:
            break
        yield user_input


def read_conversation_from(file: str) -> tuple[str]:
    """
    Reads a conversation between a Human and AI from a text file.

    Note:
        Requires text file to have line-separated conversations pre-fixed
        with either `User: ` or `LLM: `.
    Returns:
        A sequence of dialogs in the following format
        `(<user prompt>, <llm response>, ...)`
    """

    conversations = ()
    with open(file, "r") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # Verify not blank or empty
                conversations += (stripped_line,)
    return conversations


def update_memory_context(
    memory: ConversationBufferMemory, conversations: tuple[str]
) -> None:
    """
    Writes and saves `conversations` into the passed `memory` context.


    Params:
        memory (ConversationBufferMemory): Memory instance to update with conversation.
        conversations (tuple[str]): A sequence of formatted dialogs between a user and LLM.
    """

    for n in range(0, len(conversations), 2):
        if conversations[n].startswith("User"):
            prompt = conversations[n].removeprefix("User: ")
            response = conversations[n + 1].removeprefix("LLM: ")

        memory.save_context({"input": prompt}, {"output": response})


def setup_for_query(memory: ConversationBufferMemory) -> None:
    """
    Setup model to respond to questions based on conversations in context.

    Note:
        Prefer this util over `ChatPromptTemplate` to tailor the models
        response to reduce token usage on subsequent user prompts.
    """

    prompt = """
    I'd like to ask you questions.
    Give simple, non-verbose answers that respond accurately in a single sentence.
    If you don't know the answer, kindly state you can't remember.
    Use the same tone in previous conversions.
    """
    response = "Absolutely, let's get started."
    memory.save_context({"input": prompt}, {"output": response})
