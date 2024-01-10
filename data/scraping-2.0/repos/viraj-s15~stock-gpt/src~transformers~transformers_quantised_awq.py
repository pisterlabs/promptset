# Managing Imports
import os
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.tools import Tool, tool
from pydantic import BaseModel, Field
import sys
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)


try:
    sys.path.append("../../src/")
    from tools.stock_business_info_tools import stock_business_tools
    from tools.stock_data_tools import stock_data_tools
except ImportError as e:
    logging.error(f"Import error: {e}")
    stock_business_tools = None
    stock_data_tools = None

# setup import from env variables
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path)
    huggingface_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    login(token=huggingface_access_token)
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--use_flash_attention", action="store_true")
parser.add_argument("--temperature", type=float, default=0.01)
parser.add_argument("--model", type=str, default="TheBloke/zephyr-7B-alpha-AWQ")
parser.add_argument("--max_iterations", type=int, default=3)
parser.add_argument("--message_history", type=str, default=5)
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--repetition_penalty", type=float, default=1.1)

args = parser.parse_args()

verbose = False
if args.verbose:
    verbose = True

flash_attention = False
if args.use_flash_attention:
    flash_attention = True

logging.info("Loading model and tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    args.model, cache_dir=args.cache_dir, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model, cache_dir=args.cache_dir, use_flash_attention2=flash_attention
)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    # Langchain needs full text
    return_full_text=True,
    task="text-generation",
    temperature=args.temperature,
    do_sample=True,
    max_new_tokens=args.max_new_tokens,
    repetition_penalty=args.repetition_penalty,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

tools = stock_data_tools + stock_business_tools

# This is where we store the conversation history
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=args.message_history, return_messages=True
)

agent = initialize_agent(
    # This is the only only agent which supports multi-input structured tools
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=verbose,
    max_iterations=args.max_iterations,
    early_stopping_method="generate",
    memory=conversational_memory,
    handle_parsing_errors="Check your output and make sure it conforms!",
)


print("Welcome to the stock market chatbot!")
print("Type 'exit' to exit the program.")
print("Type 'help' to see a list of flags.")
print("Remember: Yahoo Finance is a bit slow at times so please be patient")
while True:
    user_input = input(">>> ")
    if user_input == "help":
        print(
            "Flags:\n"
            + "--verbose: Sets langchain output to verbose\n"
            + "--temperature: The temperature of the model\n"
            + "--model: The model to use\n"
            + "--max_iterations: The maximum number of iterations the model is allowed to make\n"
            + "--message_history: The number of messages to store in the conversation history"
        )
        continue
    if user_input == "exit":
        break
    response = agent(user_input)
    print(response["output"])
