from llama_index.agent import OpenAIAssistantAgent
from dotenv import load_dotenv, find_dotenv
import os
import openai

# set the OPENAI_API_KEY before calling the method.
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def custom_ai_assistant():
    agent = OpenAIAssistantAgent.from_new(
        name="Math Tutor",
        instructions="You are an engineering advanced math tutor. Write and run code to answer math questions.",
        openai_tools=[{"type": "code_interpreter"}],
        instructions_prefix="Please address the user as Pavan Mantha."
    )

    response = agent.chat(
        "find the rank of the given 3x3 matrix [[3, 6, 7], [4, 7, 8], [6, 5, 8]]. Can you help me ?"
    )

    print(str(response))


custom_ai_assistant()
