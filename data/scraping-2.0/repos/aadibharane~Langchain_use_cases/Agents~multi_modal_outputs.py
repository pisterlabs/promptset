#Multi-modal outputs: Image & Text
'''
This notebook shows how non-text producing tools can be used to create multi-modal agents.
This example is limited to text and image outputs and uses UUIDs to transfer content across tools and agents.
This example uses Steamship to generate and store generated images. Generated are auth protected by default.
You can get your Steamship api key here: https://steamship.com/account/api
'''
# from steamship import Block, Steamship
# import re
# from IPython.display import Image
# import os
# os.environ["OPENAI_API_KEY"] ="api key"
# #serpapi_key="serpapi_key"

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool

# llm = OpenAI(temperature=0)
# #Dall-E
# tools = [
#     SteamshipImageGenerationTool(model_name= "dall-e")
# ]

# mrkl = initialize_agent(tools, 
#                         llm, 
#                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#                         verbose=True)

# output = mrkl.run("How would you visualize a parot playing soccer?")

# from steamship import Block, Steamship
# import re
# from IPython.display import Image
# import os
# os.environ["OPENAI_API_KEY"] ="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"
# os.environ['STEAMSHIP_API_KEY']="9A04BC86-CD0D-479E-B448-E245B19BC9AB"

# from langchain import OpenAI
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.tools import SteamshipImageGenerationTool

# llm = OpenAI(temperature=0)

# tools = [
#     SteamshipImageGenerationTool(model_name="dall-e")
# ]

# for tool in tools:
#     tool.update_forward_refs()

# SteamshipImageGenerationTool.update_forward_refs()  # Call update_forward_refs() for SteamshipImageGenerationTool

# mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# output = mrkl.run("How would you visualize a parrot playing soccer?")

from steamship import Block, Steamship
import re
from IPython.display import Image
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ['STEAMSHIP_API_KEY'] = "STEAMSHIP_API_KEY"

from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import SteamshipImageGenerationTool

llm = OpenAI(temperature=0)

tools = [
    SteamshipImageGenerationTool(model_name="dall-e")
]

for tool in tools:
    tool.update_forward_refs()

SteamshipImageGenerationTool.update_forward_refs()  # Call update_forward_refs() for SteamshipImageGenerationTool

mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

output = mrkl.run("How would you visualize a parrot playing soccer?")

