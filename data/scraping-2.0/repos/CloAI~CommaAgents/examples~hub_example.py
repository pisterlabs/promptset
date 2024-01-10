from comma_agents.hub.agents.cloai.openai import OpenAIAPIAgent
from comma_agents.hub.agents.cloai.llama_cpp import LLaMaAgent

agent = OpenAIAPIAgent("Open AI Agent", config={
    "model_name": "gpt-4",
})

llama_agent = LLaMaAgent("LLaMa Agent", llama_config={
    "model_path": "{mod_file}"
})

llama_agent.call("How are you doing?")