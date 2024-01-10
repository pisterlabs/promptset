"""
File that contains the custome LangChain tool process_thoughts.
"""
import logging

from langchain.agents import tool

@tool
def process_thoughts(thought: str) -> str: 
    """This is useful for when you have a thought that you want to use in a task, 
    but you want to make sure it's formatted correctly. 
    Input is your thought and self-critique and output is the processed thought."""
    logging.info("Processing thought...")
    return thought