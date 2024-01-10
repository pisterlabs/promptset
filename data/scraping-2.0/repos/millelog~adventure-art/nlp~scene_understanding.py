# nlp/scene_understanding.py
from openai import OpenAI
import json
from database import DatabaseManager
from nlp.summarization import Summarizer
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from utils.utils import scene_to_text
# Initialize your database manager


class scene_processor:
    def __init__(self, db_manager):
        self.summarizer = Summarizer(db_manager)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.db_manager = db_manager

    def process_scene_text(self, text: str):
        """Process the text to understand the scene and update the database."""

        functions = [
            {
                "name": "create_new_scene",
                "description": "Create a new scene entry in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene_description": {"type": "string"},
                        "characters_present": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["scene_description", "characters_present", "characters_descriptors"]
                }
            },
            {
                "name": "update_current_scene",
                "description": "Update the current scene entry in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene_title": {"type": "string"},
                        "scene_description": {"type": "string"},
                        "characters_present": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["scene", "scene_title", "scene_description", "characters_present", "characters_descriptors"]
                }
            }
        ]

        messages = [
            {"role": "system", "content": "You are a helpful assistant for understanding scenes and characters. only respond with defined functions."},
            {"role": "user", "content": f"Process the following text to understand the scene and characters: {text}"}
        ]

        current_scene = self.db_manager.get_current_scene()

        if current_scene:
            messages.append(
                {"role": "system", "content": f"The most recent scene object is: {scene_to_text(current_scene)}"})

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            functions=functions,
        )

        # Updating the response handling to work with the new Pydantic model
        response_message = response.choices[0].message

        # Checking if the response message has a 'function_call'
        if hasattr(response_message, "function_call") and response_message.function_call:
            function_name = response_message.function_call.name
            # Parsing the arguments from JSON string to dictionary
            function_args = json.loads(
                response_message.function_call.arguments)

            if function_name == "create_new_scene":
                # Summarize the ended scene
                if current_scene:
                    summary = self.summarizer.summarize_scene(current_scene)
                    
                return self.db_manager.create_new_scene(**function_args)

            elif function_name == "update_current_scene":
                return self.db_manager.update_current_scene(**function_args)


if __name__ == "__main__":
    text_sample = "In a dark and stormy night, Arthur and Merlin were devising a plan in the grand hall."
    # Assume this method gets the latest scene from the database
    scene_processor_instance = scene_processor()
    scene_processor_instance.process_scene_text(text_sample)
