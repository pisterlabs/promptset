import os

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import SteamshipImageGenerationTool

# Load environment variables from .env file
load_dotenv()
os.environ["STEAMSHIP_API_KEY"] = '843361DB-40E8-4058-B329-04640C49F5EC'

llm = OpenAI(temperature=0)
# tools = [
#     SteamshipImageGenerationTool(model_name="dall-e")
# ]

tools = [
    SteamshipImageGenerationTool(model_name="stable-diffusion")
]

mrkl = initialize_agent(tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True)

output = mrkl.run("How would you visualize a parot playing soccer?")

print(output)
