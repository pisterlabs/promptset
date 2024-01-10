
###############################################################################
# Weekly Metageopolitical PESTLE report
# Written by Kan Yuenyong (kan.yuenyong@siaintelligenceunit.com)
# Version 0.1
# Wed Jun 14 10:57:08 UTC 2023
#
#
#

import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain import PromptTemplate
from langchain.llms import OpenAI


# Define the filename
filename = 'revised_accu_process_tweet.txt'

# Check if file exists
if os.path.exists(filename):
    # Open the file
    with open(filename, 'r') as file:
        # Read the data
        data = file.readlines()

    # Now `data` is a list where each element is a line from your file
else:
    print(f"The file {filename} does not exist.")




#----------- PART1: ASK GPT to select tweet ------
#
# Initialize the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

template = """
You are an expert in geopolitical analysis specialized in metageopolitics (explained in metageopolitical knowledge graph section) and are tasked with creating a compelling, well-structured, and informative weekly essay (in weekly scan section) for C-suite executives. The essay should cover the most critical priorities grouped into PESTLE (Political, Economic, Sociocultural, Technological, Legal, and Environmental) sections. Your goal is to provide concise yet comprehensive insights, allowing executives to make informed decisions for their organizations.

Context: {context}

Question: {query}

Answer: """


# Define the context variable
context = """
Metageopolitical knowledge graph:
Overview
An integrative framework that combines various geopolitical theories
Seeks to address shortcomings and limitations of individual theories
Draws inspiration from Alvin Toffler's "powershift" concept

Powershift
    Foundation
        Inspired by The Three Sacred Treasures of Japan
            Valor (hard power)
            Wisdom (noopolitik)
            Benevolence (economic power)
        Recognizes the dynamic interplay of different powers in various domains

Geopolitical Theories
    Heartland Theory (Sir Halford John Mackinder)
        Emphasizes the strategic significance of controlling the central landmass of Eurasia
    Rimland Theory (Nicholas John Spykman)
        Highlights the importance of controlling coastal regions for geopolitical advantage
    Geopolitical Implications and Hard Power (George Friedman)
        Expands upon the Heartland and Rimland theories, accounting for modern geopolitical realities
    Offensive Realism (John Joseph Mearsheimer)
        Concentrates on the pursuit of regional hegemony as a primary goal in international politics
    Neoliberalism
        Stresses the role of global institutions and economic power in shaping international relations
    Constructivism
        Views institutions as the result of human interactions and the construction of shared ideas

Metageopolitical Framework Applications
    Inclusive Approach
        Integrates insights from multiple schools of thought for a more comprehensive understanding
    Multidimensional Analysis
        Takes into account military, economic, political, and social factors in assessing geopolitical situations
    Universal Application
        Adaptable to a wide range of international relations scenarios, enabling better predictions and strategic decisions

        Analyze the news article in terms of the following metageopolitical aspects:

- Hard power dynamics
- Economic power influences
- Noopolitik elements (information and ideas)
- State actors' roles and motivations
- Non-state actors' roles and motivations

While analyzing the news article, consider the broader implications of the events and their impact on global power dynamics, international relations, and potential shifts in the balance of power.

Provide a summary of the news article, highlighting the key insights from the metageopolitical analysis and its potential implications on global power dynamics.
The metageopolitics model is designed to incorporate various schools of thought, such as mainstream economics and economic statecraft, and is built on the foundation of dynamic statecraft and "positive governance." As an ongoing research effort, this framework aims to refine and enhance its capacity to analyze and interpret geopolitical intricacies.

Weekly scan section:
"""

# Read query from a file
#with open('/tmp/jj.txt', 'r') as file:
#    query = file.read().split("\n")

query = data
    
# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

# Create a new essay by formatting the context and query parameters with the PromptTemplate object
newessay = prompt_template.format(context=context, query=query)

# Initialize an OpenAI object
openai_instance = OpenAI(
    model_name="gpt-4",
    openai_api_key=openai.api_key
)

# Generate a response from the OpenAI API
response = openai_instance(newessay)

# Print the response to the console
#print(response)

# Print the response to safe into file
with open('/tmp/superb.txt', 'w') as file:
    file.write(str(response))
