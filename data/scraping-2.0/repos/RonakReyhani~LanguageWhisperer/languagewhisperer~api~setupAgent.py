from huggingface_hub import login
import getpass
from transformers.tools import HfAgent
from transformers.tools import OpenAiAgent
from searchWikiTool import search_tool
# login to huggingface hub
# TODO: remove the token
token = "put_your_token_here"

def config_agent():
  login(token)
  agent_name = "StarCoder (HF Token)" #@param ["StarCoder (HF Token)", "OpenAssistant (HF Token)", "OpenAI (API Key)"]
  if agent_name == "StarCoder (HF Token)":
      agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[search_tool])
      print("StarCoder is initialized 💪")
      return agent
  elif agent_name == "OpenAssistant (HF Token)":
      agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
      print("OpenAssistant is initialized 💪")
      return agent
  if agent_name == "OpenAI (API Key)":
      pswd = getpass.getpass('OpenAI API key:')
      agent = OpenAiAgent(model="text-davinci-003", api_key=pswd)
      print("OpenAI is initialized 💪")
      return agent