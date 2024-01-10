"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `eval.py`.
We've included a few to get your started."""

import logging
import re
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo
from rank_bm25 import BM25Okapi as BM25_Index
from openai import OpenAI

client = OpenAI()

# Use the same logger as agents.py, since argubots are agents;
# we split this file
# You can change the logging level there.
log = logging.getLogger("agents")

#############################
# Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
# Other argubot classes and instances -- add your own here!
############################################################


class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""

    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(
                previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(
                f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")

            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])

        return claim

# Akiko doesn't use an LLM, but looks up an argument in a database.


# get the Kialo database from text files
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))


###########################################
# Define your own additional argubots here!
###########################################
class AkikiAgent(Agent):
    """ AkikiAgent subclasses the Agent class. It responds by looking back 3 dialogues and
    weighting more recent turns heavier than earlier turns. It uses a Kialo database."""

    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            claim = self.kialo.random_chain()[0]
        else:
            # Look back 3 dialogues or as many as available
            num_dialogues_to_look_back = min(3, len(d) - 1)

            # Separate human turns and Akiki's turns
            human_turns = [d[-i-1]['content']
                           for i in range(num_dialogues_to_look_back) if d[-i-1]['speaker'] != "Akiki"]
            akiki_turns = [d[-i-1]['content']
                           for i in range(num_dialogues_to_look_back) if d[-i-1]['speaker'] == "Akiki"]

            # Adjust weights based on whether it's a human turn or Akiki's turn
            if not akiki_turns:  # If no Akiki turns found, consider it's a human turn
                weights = [4, 3, 2]  # Adjust weights for human turns
                recent_turns = human_turns
            else:
                weights = [3, 2, 1]  # Adjust weights for Akiki's turns
                recent_turns = akiki_turns

            # Weight more recent turns heavier than earlier turns
            weighted_turns = [turn * weight for turn,
                              weight in zip(recent_turns, weights)]

            # Combine the weighted turns to form the input for similarity comparison
            input_text = ' '.join(weighted_turns)

            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(
                input_text, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(
                f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")

            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])

        return claim


akiki = AkikiAgent("Akiki", Kialo(glob.glob("data/*.txt")))

# aragorn


class RAGAgent(Agent):
    """ RAGAgent subclasses the Agent class. It responds by looking back 3 dialogues and
    weighting more recent turns heavier than earlier turns. It uses a Kialo database."""

    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            claim = self.kialo.random_chain()[0]
        else:
            # Look back 3 dialogues or as many as available
            num_dialogues_to_look_back = min(3, len(d) - 1)

            # Separate human turns and Akiki's turns
            human_turns = [d[-i-1]['content']
                           for i in range(num_dialogues_to_look_back) if d[-i-1]['speaker'] != "Akiki"]
            akiki_turns = [d[-i-1]['content']
                           for i in range(num_dialogues_to_look_back) if d[-i-1]['speaker'] == "Akiki"]

            # Adjust weights based on whether it's a human turn or Akiki's turn
            if not akiki_turns:  # If no Akiki turns found, consider it's a human turn
                weights = [4, 3, 2]  # Adjust weights for human turns
                recent_turns = human_turns
            else:
                weights = [3, 2, 1]  # Adjust weights for Akiki's turns
                recent_turns = akiki_turns

            # Weight more recent turns heavier than earlier turns
            weighted_turns = [turn * weight for turn,
                              weight in zip(recent_turns, weights)]

            # Combine the weighted turns to form the input for similarity comparison
            input_text = ' '.join(weighted_turns)

            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(
                input_text, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(
                f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")

            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])

        return claim


aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")))


class RAGAgent_dense(Agent):
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            # No context available, generate a random response or use default behavior
            return "I don't have enough information to provide a response at the moment."
        else:
            # Collect recent turns from the dialogue
            num_dialogues_to_look_back = min(3, len(d) - 1)
            recent_turns = [d[-i-1]['content'] for i in range(num_dialogues_to_look_back)]

            # Combine recent turns into a single string
            input_text = ' '.join(recent_turns)

            # Use the OpenAI ChatGPT API with the knowledge retrieval tool
            response = client.beta.assistants.create(
                model="gpt-4.0-turbo",
                tools=["knowledge-retrieval"],
                messages=[
                    {"role": "system", "content": "You are a customer support chatbot. Use your knowledge base to best respond to customer queries."},
                    {"role": "user", "content": input_text}
                ]
            )

            # Extract the model-generated reply from the API response
            reply = response['choices'][0]['message']['content']

            return reply
    
rag_dense = RAGAgent_dense("Rag_dense", Kialo(glob.glob("data/*.txt")))
