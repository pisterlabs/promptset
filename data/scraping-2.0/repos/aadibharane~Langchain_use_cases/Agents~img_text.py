# from steamship import Block, Steamship
# import re
# from IPython.display import Image
# import os
# os.environ['OPENAI_API_KEY'] = 'your_api_key'
# os.environ['STEAMSHIP_API_KEY'] = "STEAMSHIP_API_KEY"

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool
# from pydantic import BaseModel, Field
# llm = OpenAI(temperature=0)

# # DALL-E
# class SteamshipImageGenerationTool(BaseModel):
#     model_name: str = Field(default="dall-e", alias="model-name")
#     is_single_input: bool = True
#     description: str = ""
#     name: str = ""

# SteamshipImageGenerationTool.update_forward_refs()

# tools = [
#     SteamshipImageGenerationTool()
# ]

# mrkl = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# output = mrkl.run("How would you visualize a parrot playing soccer?")
# print(output)


# from steamship import Block, Steamship
# import re
# from IPython.display import Image
# import os
# os.environ['OPENAI_API_KEY'] = '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'
# os.environ['STEAMSHIP_API_KEY'] = "9A04BC86-CD0D-479E-B448-E245B19BC9AB"

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool
# from pydantic import BaseModel, Field

# llm = OpenAI(temperature=0)

# # DALL-E
# class SteamshipImageGenerationTool(BaseModel):
#     model_name: str = Field(default="dall-e", alias="model-name")
#     is_single_input: bool = True
#     description: str = ""
#     name: str = ""

# SteamshipImageGenerationTool.update_forward_refs()

# tools = {
#     'tools': [
#         SteamshipImageGenerationTool()
#     ]
# }

# mrkl = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# output = mrkl.run("How would you visualize a parrot playing soccer?")
# print(output)



#################################

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool
# from pydantic import BaseModel, Field
# import requests
# from PIL import Image
# from io import BytesIO
# import os

# # Set up OpenAI API and Steamship API keys
# openai_api_key = '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'
# steamship_api_key = "9A04BC86-CD0D-479E-B448-E245B19BC9AB"

# openai_api_key = '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'
# os.environ['OPENAI_API_KEY'] = openai_api_key

# os.environ['STEAMSHIP_API_KEY'] =steamship_api_key
# # Initialize OpenAI language model
# llm = OpenAI(api_key=openai_api_key, temperature=0.5)

# class SteamshipImageGenerationTool(BaseModel):
#     model_name: str = Field(default="dall-e", alias="model-name")
#     is_single_input: bool = True
#     description: str = ""
#     name: str = ""

# SteamshipImageGenerationTool.update_forward_refs()


# # Create a list of tools with SteamshipImageGenerationTool
# tools = {
#     'tools': [
#         SteamshipImageGenerationTool()
#     ]
# }

# # Initialize the agent
# agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# # Generate the multi-modal output
# prompt = "How would you visualize a parrot playing soccer?"
# output = agent.run(prompt)

# # Extract the image URL and display the image
# image_url = output["tools_output"]["image"]["content"]
# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content))
# image.show()

# # Print the generated text description
# description = output["tools_output"]["description"]["content"]
# print("Generated Description:", description)

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool
# from pydantic import BaseModel, Field
# import requests
# from PIL import Image
# from io import BytesIO
# import os

# #Set up OpenAI API and Steamship API keys
# openai_api_key = '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'
# steamship_api_key = "9A04BC86-CD0D-479E-B448-E245B19BC9AB"

# os.environ['OPENAI_API_KEY'] = openai_api_key

# os.environ['STEAMSHIP_API_KEY'] = steamship_api_key

# #Initialize OpenAI language model
# llm = OpenAI(api_key=openai_api_key, temperature=0.5)

# class SteamshipImageGenerationTool(BaseModel):
#     model_name: str = Field(default="dall-e", alias="model-name")
#     is_single_input: bool = True
#     description: str = ""
#     name: str = ""

# SteamshipImageGenerationTool.update_forward_refs()

# #Create a list of tools with SteamshipImageGenerationTool
# tools = [{
#     'tool': SteamshipImageGenerationTool()
# }]

# #Initialize the agent
# agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# #Generate the multi-modal output
# prompt = "How would you visualize a parrot playing soccer?"
# output = agent.run(prompt)

# #Extract the image URL and display the image
# image_url = output["tools_output"]["image"]["content"]
# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content))
# image.show()

# #Print the generated text description
# description = output["tools_output"]["description"]["content"]
# print("Generated Description:", description)

#@title Setup
transformers_version = "v4.29.0" #@param ["main", "v4.29.0"] {allow-input: true}

print(f"Setting up everything with transformers version {transformers_version}")

#pip install huggingface_hub>=0.14.1 git+https://github.com/huggingface/transformers@$transformers_version -q diffusers accelerate datasets torch soundfile sentencepiece opencv-python openai

import IPython
import soundfile as sf

def play_audio(audio):
    sf.write("speech_converted.wav", audio.numpy(), samplerate=16000)
    return IPython.display.Audio("speech_converted.wav")

from huggingface_hub import notebook_login
notebook_login()

#@title Agent init
agent_name = "OpenAI (API Key)" #@param ["StarCoder (HF Token)", "OpenAssistant (HF Token)", "OpenAI (API Key)"]

import getpass

if agent_name == "StarCoder (HF Token)":
    from transformers.tools import HfAgent
    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    print("StarCoder is initialized 💪")
elif agent_name == "OpenAssistant (HF Token)":
    from transformers.tools import HfAgent
    agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
    print("OpenAssistant is initialized 💪")
if agent_name == "OpenAI (API Key)":
    from transformers.tools import OpenAiAgent
    pswd = getpass.getpass('OpenAI API key:')
    agent = OpenAiAgent(model="text-davinci-003", api_key=pswd)
    print("OpenAI is initialized 💪")

boat = agent.run("Generate an image of a boat in the water")
print(boat)