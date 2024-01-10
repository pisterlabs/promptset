import os
import sys
import textwrap
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from langchain import HuggingFaceHub, LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain.chains.summarize import load_summarize_chain

# Add the path to the directory containing your_script.py
script_dir = os.path.dirname(os.path.abspath(__file__))
two_folders_up = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(two_folders_up)

llm_agent_tools = [
    "document-question-answering",
    "text-question-answering",
    "text-to-speech",
    "huggingface-tools/text-download",
    "context_analysis",
    "knowledge_retrieval",
    "text_generation",
    "data_processing",
    "conversational_logic",
    "task_execution",
    "content_generation",
    "user_interaction",
    "contextual_decision_making",
    "human", 
    "llm-math"
]

hf_tools = load_tools(llm_agent_tools)

llm: LLM = G4FLLM(
    model=models.gpt_35_turbo,
)

agent_chain = initialize_agent(
    hf_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

question = "How do I make a sandwich?"
response = agent_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)

 
