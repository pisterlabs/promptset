# Copyright (C) 2023 Adrien Baumert
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Project: GPT-Interactive-Text-Session
# File: gpt_agent.py
# Version : 1.0.0
# =================================================

# Importing OpenAI
import openai

# Importing get_keys from retrieving_keys
from get_keys import retrieving_keys

# Retrieving OpenAI Organization ID and OpenAI GPT API Key from .env file
keys = retrieving_keys()

# Set organization ID and API key for the OpenAI API
openai.organization = keys["org_id"]
openai.api_key = keys["api_key"]

# Running a check to ensure api key is correct
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        # Running check
        {"role": "system", "content": "TEST"}
    ]
)

class GPTAgent:
    # A class to interact with the GPT Agent and manage its memory
    def __init__(self):
        # Initialize GPTAgent by loading initial memory from a text file
        with open('gpt_agent_initial_memory.txt', 'r') as initial_memory:
            self.memory = initial_memory.read()
            self.memory = self.memory

            # Printing the initial GPT-Agents memory the first time it is instantiated
            print(self.memory)

    def generate_response(self, prompt, memory):
        # Generate a response from the GPT Agent given a memory and a prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # Generate a response from the GPT agent based on the input prompt and memory

                # Ensuring GPT Agent understands "AI-Language Model: " prefix formatting
                {"role": "system", "content": "The following is a convorsation between a user and a AI-Language model. The AI-Language model and must send the ""AI-Language Model: "" prefix."},

                # Provide the agent's memory as a system message
                {"role": "system", "content": memory},

                # Provide the user's input as a user message
                {"role": "user", "content": prompt},
            ]
        )
        # Return only the content of the response
        return response.choices[0].message['content'].strip()

    def append_gpt_agent_memory(self, memory):
        # Append new memory to the existing memory
        self.memory += memory

    def get_memory(self):
        # Retrieve the current memory of the GPT Agent
        return self.memory