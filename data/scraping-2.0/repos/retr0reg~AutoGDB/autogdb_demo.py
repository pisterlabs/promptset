#encoding: utf-8
"""
Demo for the unpublished package "AutoGDB"
This demo showcases how to use the "AutoGDB" package to connect ChatGPT with GDB for dynamic debugging tasks.

@author: retr0@retr0.blog
@description: Use ChatGPT for dynamic-debugging with GDB
"""

# Import all necessary modules from the AutoGDB package.
from autogdb import AutoGDB, PwnAgent

# Import API keys from a secure location.
# Make sure to create an api_key.py file with your personal OPENAI_API_KEY and OPENAI_API_BASE variables.
import os
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", default="https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY == None:
    try:
        from api_key import OPENAI_API_BASE, OPENAI_API_KEY
    except ImportError:
        print("Please create an api_key.py file with your personal OPENAI_API_KEY and OPENAI_API_BASE variables or set the environment variables.")
    exit(1)

# Main execution point of the script.
if __name__ == "__main__":
    # Step 1: Establish a connection with the GDB server
    # Replace 'localhost' and '5000' with your server's IP address and port number.
    # The tool() method will create a "Tool" class instance that is compatible with langchain.
    autogdb_connection = AutoGDB("localhost", "5000").tool()

    # Step 2: Create a PwnAgent instance
    # The PwnAgent uses the provided Tool class instance to interact with the GDB server.
    # It contains a pre-built zero-shot-ReAct Agent to assist with solving challenges logically.
    # If you don't have a specific API_BASE, use the default OpenAI API endpoint.
    default_api_base = "https://api.openai.com/v1"
    pwnagent = PwnAgent(OPENAI_API_KEY, OPENAI_API_BASE or default_api_base, autogdb_connection)

    # Step 3: Start interacting with the PwnAgent
    # Now, you can give commands to the PwnAgent to perform debugging tasks!!!!
    response = pwnagent.chat("Find what this code does")
    print(response)

# This demo is now complete! You can expand upon it by adding more functionality or creating more complex interactions.
