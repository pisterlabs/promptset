from openai import OpenAI
from instructor import patch, OpenAISchema
from pydantic import Field
import os
from dotenv import load_dotenv
from ai.system_prompt_builder import SystemPromptBuilder

##################################
############ SCHEMA ##############
##################################

"""
Rather than hoping the correctly structured data is in the response, this script takes a fully structured approach to prompting.
OpenAI's JSON mode is unreliable, necessitating the use of the `instructor` library with `pydantic`.
Using vanilla OpenAI API with function calling and JSON mode, the prompting structure is `(prompt: str, schema: dict) -> str`
The above implementation doesn't guarantee that the desired data is in the response object.
A far more reliable approach is `(prompt: str, schema: Model) -> Model`, allowing for validation of fields.
"""

# NOTE: All content within the schema gets sent to OpenAI's API.
# NOTE: The schema should only contain information that is relevant to the prompt, and nothing else.
# NOTE: The docstrings, attributes, types, and field descriptions are all of equal importance for the prompt.


# Sub-schema for player options
class PlayerOptions(OpenAISchema):
    """
Player options for a scenario in Immemoria.
One action should be Order, one should be Chaos, and one should be Neutral.
Actions vary significantly from previous scenarios.
    """
    order: str = Field(..., description="Order action option for the player.")
    chaos: str = Field(..., description="Chaos action option for the player.")
    neutral: str = Field(..., description="Neutral action option for the player.")


# Main response schema
class Scenario(OpenAISchema):
    """
Defines a scenario in the game Immemoria, including the description, player actions, and a summary.
The description is a succinct description of the current scenario in Immemoria that the player must react to.
The player does not always get a favorable outcome for the description.
Each `Scenario` is no more than 3 sentences.
Each `Scenario` has 3 potential `PlayerOptions` in each scenario.
    """
    description: str = Field(..., description="Three-sentence description of the current scenario in Immemoria.")
    actions: PlayerOptions = Field(..., description="Options available to the player in the scenario.")
    summary: str = Field(..., description="One-sentence summary of the PLAYER's action and its result.")


##################################
######### Main AI Class ##########
##################################

class ImmemoriaAI():
    """
    A class for generating responses for Immemoria.
    """
    def __init__(self):
        self.model = "gpt-4-1106-preview" # gpt-3.5-turbo-1106
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = patch(self.client)
        ##### TEMPORARY VARIABLES #####
        self.conversation_history = []
        self.summary = []
        
    def generate_response(self, player_prompt: str):
        system_prompt = SystemPromptBuilder.construct_default_gameplay_loop_prompt(self.conversation_history, self.summary)
        # Send request to OpenAI with structured system prompt and response model
        response: Scenario = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_model=Scenario,
            temperature=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": player_prompt},
            ],
        )
        self.add_to_summary(response.summary + " ")
        response_with_full_summary = response.model_dump()
        response_with_full_summary["summary"] = self.get_summary()
        return response_with_full_summary

    ################# TEMPORARY FUNCTIONS #################
    # TODO: These will be handled through the database in immemoria_instructor_v3.PlayerOptions

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

    def add_to_summary(self, sentence):
        """
        Adds detail about the current interaction to the summary

        Takes in a prompt and response, sends them to the OpenAI API with the
        current summary, and returns the new summary.

        The summary should never exceed two paragraphs.
        """
        self.summary.append(sentence)
        # Remove the oldest entry if the summary is too long
        if len(self.summary) > 10:
            self.summary.pop(0)

    def get_summary(self):
        """
        Returns the summary.
        """
        return self.summary

    def clear_conversation_history(self):
        """
        Clears the conversation history.
        """
        self.conversation_history = []

    def clear_summary(self):
        """
        Clears the summary.
        """
        self.summary = []