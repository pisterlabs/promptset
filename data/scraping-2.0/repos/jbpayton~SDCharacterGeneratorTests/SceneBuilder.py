import json
import os
import sys

from CharacterBuilder import CharacterBuilder
from VNImageGenerator import VNImageGenerator

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class SceneBuilder:
    def __init__(self, chat_llm=None):
        if chat_llm is None:
            self.chat_llm = ChatOpenAI(
                model_name='gpt-4',
                temperature=0.0
            )
        else:
            self.chat_llm = chat_llm

    visual_novel_scripting_prompt = """
        As a script generator for a visual novel, your task is to create JSON-formatted messages that dictate the flow and 
        elements of the story. Each message type has specific fields that must be included for accurate representation 
        in the game engine. Follow these guidelines and format specifications carefully (for example 'happy' is not a valid
        expression for a character, but 'smile' is).

        Message Types and Formats:

        Change Scene:
           - Format: {"type": "change_scene", "location_name": "<Reusable location name>", "location_description": "<detailed description of image to be used as a background for the scene>"}
        Change to CG (Computer Graphics):
           - Format: {"type": "change_cg", "description": "<detailed description of the scene to be used as a SD prompt>"}
        Add Character (you must use only the expressions listed below):
           - Format: {"type": "add_character", "character_name": "<name>", "expression": "<cry/angry/smile/neutral/disappointed/blush/scared/laugh/yell>", "facing_direction": "<left/right>", "position": "<left/center/right>"}

        Remove Character:
           - Format: {"type": "remove_character", "position": "<left/center/right>", "transition": "<transition_type>"}
        Dialog:
           - Format: {"type": "dialog", "text": "<dialog_text>", "speaker": "<speaker_name>", "displayed_speaker": "<display_name>"}
        Narrator Text:
           - Format: {"type": "event_text", "description": "<test giving narrator and internal monologue of the player character>"}
        Question to Player:
           - Format: {"type": "question", "text": "<question_text>", "options": ["<option1>", "<option2>", ...]}

        Remember to:
        - Maintain narrative coherence and logical consistency in your messages.
        - Ensure that transitions, character placements, and scene changes align with the story's flow.
        - Format the output as JSON objects as specified for each message type.
        - The whole output should be a list of JSON objects, each representing a message.

        Generate a visual novel script based on the above specifications for a given storyline or input.
        """

    def generate_scene(self, scene_prompt, character_list, background_information=None):
        # the character list is a list of dictionaries, convert it to a JSON string
        character_list_json = json.dumps(character_list)

        # if there is no background information, set it to an empty string
        if background_information is None:
            background_information = ""

        # generate the prompt using the scene prompt, character list, and background information
        prompt = f"Please generate a full visual novel scene. The scene is set in [{scene_prompt}]. The characters are [{character_list_json}]. {background_information}"

        message = self.chat_llm(
            [
                SystemMessage(role="VisualNovelScripter", content=SceneBuilder.visual_novel_scripting_prompt),
                HumanMessage(content=prompt),
            ]
        )

        formatted_json = message.content.strip()

        # Attempt to parse the string response into a JSON object
        try:
            structured_output = json.loads(formatted_json)
        except json.JSONDecodeError:
            # Handle the case where the response is not in proper JSON format
            structured_output = "The AI's response was not in a valid JSON format. Please check the AI's output."

        return structured_output


if __name__ == "__main__":
    from util import load_secrets

    load_secrets()

    character_list = []

    # Test the CharacterBuilder class
    character_names = ["Hoshi_Yumeko", "Hikari_Yumeno", "Aiko_Tanaka"]

    #each character has a folder under the output folder, with a prompt.json file, load these to a list
    for character_name in character_names:
        with open(f"output/{character_name}/prompt.json", "r") as f:
            character_prompt = json.load(f)
        character_list.append(character_prompt)

    # Test the SceneBuilder class
    SceneBuilder = SceneBuilder()
    scene_prompt = "The scene is set in a zoo on a sunny day. The characters are talking about how cute the animals are."
    background_information = ""
    structured_output = SceneBuilder.generate_scene(scene_prompt, character_list, background_information)

    print(structured_output)

    #save the output to a file
    with open("output/scene.json", "w") as f:
        json.dump(structured_output, f, indent=4)



