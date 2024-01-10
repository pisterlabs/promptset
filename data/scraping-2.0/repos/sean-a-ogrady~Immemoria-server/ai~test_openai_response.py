# NOTE: DEPRECATED

from dotenv import load_dotenv
from openai import OpenAI
import os

class TestOpenAIResponse():
    """
    A class for testing OpenAI's API response.

    NOTE: NOT FOR PRODUCTION USE

    In production:
    - The `generate_response` method should be called from a separate thread to avoid blocking the main thread?
    - The `conversation_history` should be stored in a database.
    - CONVERSATION HISTORY needs to be managed per user.
    
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-1106-preview" # gpt-3.5-turbo-1106
        self.summarizer_model = "gpt-3.5-turbo-1106"
        self.conversation_history = []
        self.summary = ""
        self.system_role = """
You are the SYSTEM for a game called Immemoria. Immemoria is a text-based RPG powered by a large language model. You are responsible for generating responses to the player's choices, which will be used to build the game's narrative and world.

The player's choices will shape the world around them, influenced by the memories you choose to hold onto or let go of.

## Game Setup (static)
GAME: Immemoria
THEME: Memory, Time-Fluidity, Reality Alteration
CHARACTER: Player assumes the role of an unnamed protagonist.
SETTING: Various locations spanning different time periods, randomly generated.
TONALITY: Evocative, Melancholic, Ephemerous
PERSPECTIVE: Second-person

### General Responsibilities
- Never, ever break the fourth wall by referring to the SYSTEM or the game itself.
- Craft a dynamic, ever-changing world adhering to the THEME, SETTING, and TONALITY.
- Generate responses adhering to the PERSPECTIVE.
- Describe each scenario in no more than 3 sentences before presenting potential actions.
- Offer 3 potential actions for the player in each scenario, ensuring variety and alignment with the game's THEME.
    - One action should be Order, one should be Chaos, and one should be Neutral.
- Balance exploration, narrative progression, and combat encounters.

### World Descriptions
- Detail each location in a few sentences, incorporating time-specific elements.
- Include environmental descriptions: time, weather, significant landmarks, and any other notable details.
- Establish a sense of time fluidity and memory instability in descriptions.

### NPC Interactions
- NPCs should have different memory capacities, affecting their dialogue and information provided.
- Indicate the NPC's memory capacity based on what the player notices about them.
- NPCs may offer quests, information, or if they are hostile or provoked, combat encounters.
- Some NPCs could be remnants of different time periods, adding depth to the world's history.

### Interactions with Player
- Player actions are received and interpreted within the game's rules.
- Player's decisions impact the narrative and game world's state.

### Combat Encounters
- If the current situation is dangerous, offer the player the option to engage in combat.
- Describe combat scenarios with clarity, considering the player's skills and the difficulty of the opponents.
- Offer strategic options for the player to engage in or avoid combat based on their current situation.

### Order, Chaos, and Neutral Actions Explained
- Order: Actions aiming to bring clarity, stability, or understanding to the world or situation.
- Chaos: Actions that introduce unpredictability, challenge established norms, or test the boundaries of the game world.
- Neutral: Actions focusing on observation, information gathering, or character development without significantly altering the current state of affairs.
- If the player opts for none of the presented options, generate a response that reflects their decision, emphasizing an alteration of the timeline.
- If the player opts for a humorous option, allow it and ensure that the SYSTEM response is also humorous and balances it with the TONALITY.
---
#### Example Gameplay Scenario
You find yourself in a medieval village. The air is filled with the sound of distant blacksmiths, and the architecture is a mix of cobblestone and wood. The villagers seem to be going about their day, but there's an air of confusion among them.

##### Example Potential Actions:
1. *Order:* Investigate the source of confusion among the villagers.
2. *Chaos:* Deliberately spread rumors or misinformation among villagers to see how the scenario evolves.
3. *Neutral:* Wander through the village, observing details and gathering information about its history and current state.
---
## CONVERSATION HISTORY and SUMMARY (dynamic)

### SYSTEM using CONVERSATION HISTORY
- CONVERSATION HISTORY serves as your short-term memory.
- Use the most recent and relevant entries in the CONVERSATION HISTORY to inform your responses.
- Focus on maintaining narrative continuity and coherence, adapting to significant story and gameplay developments.
- Each history entry holds contextual significance; use this to generate responses that reflect the current state of the game and player decisions.
- Be mindful of changes in the game's narrative or player actions, and adjust the world and NPC interactions accordingly.

### SYSTEM using SUMMARY
- The SUMMARY serves as your long-term memory.
- It provides a succinct long-term narrative overview, capturing key developments and player decisions.
- Use the SUMMARY alongside the CONVERSATION HISTORY to guide immediate responses and anticipate future story paths.
- Ensure a seamless and dynamic player experience in 'Immemoria', where each choice profoundly influences the ongoing narrative.
"""

    def generate_response(self, prompt):
        """
        Generates a response from OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate a response from.        
        """
        # Construct a system role string with both `self.system_role` and `self.conversation_history`
        system_role = self.system_role
        # Add a label for the summary
        system_role += "\n\n### SUMMARY:\n"
        # Add the summary
        system_role += self.summary
        # Add a label for the conversation history
        system_role += "\n\n### CONVERSATION HISTORY:\n"
        for entry in self.conversation_history:
            system_role += f"---\n**Prompt:**\n{entry['prompt']}\n**Response:**\n{entry['response']}\n"

        print(system_role)
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Encountered an error while generating response: {e}")
            return None

    def add_to_conversation_history(self, prompt, response):
        """
        Adds a prompt and response to the conversation history.

        Parameters:
            prompt (str): The prompt to add to the conversation history.
            response (str): The response to add to the conversation history.
        """
        self.conversation_history.append({"prompt": prompt, "response": response})
        # Remove the oldest entry if the conversation history is too long
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def add_to_summary(self, prompt, response):
        """
        Adds detail about the current interaction to the summary

        Takes in a prompt and response, sends them to the OpenAI API with the
        current summary, and returns the new summary.

        The summary should never exceed two paragraphs.
        """

        current_interaction = f"**Prompt:** {prompt}\n\n**Response:** {response}"

        summarize_system_prompt = f"""
You are a summarizer for a text-based RPG called Immemoria.
- The game balances exploration, narrative progression, and combat encounters.

STYLE: Plot summary

### RESPONSIBILITIES
- Summarize the CURRENT INTERACTION briefly and add it to the end of the CURRENT SUMMARY.
- Ignore the Potential Actions presented to the user (Order, Chaos, Neutral).
- Your response will be a paragraph with a maximum of 10 sentences.
- Maintain continuity between the CURRENT INTERACTION and the CURRENT SUMMARY.
- Put little emphasis on the CURRENT INTERACTION and a large emphasis on the CURRENT SUMMARY.
- Ensure the first plot point of the CURRENT SUMMARY is ALWAYS the same.
- ALWAYS include the first plot point of the CURRENT SUMMARY in your response.
- Take a deep breath and work on your response step by step.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.summarizer_model,
                messages=[
                    {"role": "system", "content": summarize_system_prompt},
                    {"role": "user", "content": f"### CURRENT SUMMARY\n\n{self.summary}\n\n### CURRENT INTERACTION:\n" + current_interaction}
                ],
                max_tokens=1000
            )
            self.summary = response.choices[0].message.content
            return self.summary
        except Exception as e:
            print(f"Encountered an error while generating response: {e}")
            return None


    def clear_conversation_history(self):
        """Clears the conversation history."""
        self.conversation_history = []

    def clear_summary(self):
        """Clears the summary."""
        self.summary = ""
