import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

assistant = client.beta.assistants.create(
    name = "OpenAI para desarrollo esencial",
    instructions = "Eres un asistente muy útil.",
    tools=[{"type":"retrieval"}, {"type":"code_interpreter"}],
    model="gpt-4-1106-preview"
)