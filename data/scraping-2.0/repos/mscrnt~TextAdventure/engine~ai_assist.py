# engine/ai_assist.py

from utilities import normalize_name
from openai import OpenAI
from icecream import ic
import json
import re

class AIAssist:
    def __init__(self, player_sheet, world_builder):
        self.player_sheet = player_sheet
        self.world_builder = world_builder

        # Load API key from engine/secrets.json
        # You can get your API key from https://beta.openai.com/
        secrets_file = 'data/secrets.json'
        with open(secrets_file, 'r') as file:
            secrets = json.load(file)
        self.api_key = secrets.get('openai_api_key', '')

        self.client = OpenAI(api_key=self.api_key)


    def generate_ai_response(self, prompt, max_tokens=1000):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="gpt-3.5-turbo-1106",
                max_tokens=max_tokens,
                temperature=0.4,
                stop=["</HTML>"],  # Example stop sequence
                frequency_penalty=.7,  # Adjust as needed
                presence_penalty=.5   # Adjust as needed
            )
            print(f"AI generated response: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            ic(f"Error in AI response generation: {e}")
            return f"I'm having trouble understanding that. {e}"


    def handle_player_command(self, command):
        # Access world data and player sheet
        world_data = self.world_builder.world_data
        player_data = self.player_sheet

        # Step 2: Send command to GPT for interpretation
        interpret_prompt = self.construct_command_context_prompt(command, world_data, player_data)
        interpreted_command = self.generate_ai_response(interpret_prompt)
        ic(f"Interpreted command: {interpreted_command}")

        # Step 3: Process the interpreted command to determine the game action
        action_response = self.determine_action_from_interpretation(interpreted_command)

        # Step 4: Execute the action and generate a narrative response
        ic(f"Action response: {action_response}")
        ic(f"Original command: {command}")
        return self.generate_narrative_response(action_response, command)

    
    def construct_command_context_prompt(self, command, world_data, player_data):
        # Dictionary of available commands, their descriptions, and examples
        command_dictionary = {
            "move to": {
                "description": "Move the player to a specified location.",
                "examples": ["I want to move to Eldergrove Forest", "go to the Great Bazaar", "travel to the Whispering Forest"],
                "parsing_guideline": "Extract the destination name from the command and format it as '(destination) location'.",
                "command_to_perform": "move to, go to",
                "command_format": "move to <destination>, e.g., go to Eldergrove Forest"
            },
            "talk to": {
                "description": "Initiate a conversation with an NPC.",
                "examples": ["talk to the guard", "speak to the merchant", "chat with the innkeeper"],
                "parsing_guideline": "Identify the NPC to be spoken to from the command.",
                "command_to_perform": "talk to, speak to",
                "command_format": "talk to <NPC>, e.g., talk to the guard"
            },
            "take": {
                "description": "Take an item from the current location or container.",
                "examples": ["take potion", "pick up sword", "grab the key"],
                "parsing_guideline": "Identify the item and quantity to be taken from the command.",
                "command_to_perform": "take",
                "command_format": "take <item>, e.g., take potion"
            },
            "examine": {
                "description": "Examine an item or the surroundings.",
                "examples": ["examine map", "look at the statue", "inspect the chest"],
                "parsing_guideline": "Determine the item or area to be examined from the command.",
                "command_to_perform": "examine",
                "command_format": "examine <item>, e.g., examine map"
            },
            "open": {
                "description": "Open a specified container in the current location.",
                "examples": ["open chest", "unlock the door", "open the bag"],
                "parsing_guideline": "Find out which container is to be opened from the command.",
                "command_to_perform": "open",
                "command_format": "open <container>, e.g., open chest"
            },
            "close": {
                "description": "Close the currently open container.",
                "examples": ["close chest", "shut the door", "close"],
                "parsing_guideline": "Understand which container is to be closed, if specified.",
                "command_to_perform": "close",
                "command_format": "close, e.g., close chest"
            },
            "give": {
                "description": "Give an item to an NPC or place it in a container.",
                "examples": ["give potion to guard", "put 5 potion in chest"],
                "parsing_guideline": "Identify the item and recipient (NPC or container) from the command.",
                "command_to_perform": "give",
                "command_format": [
                    "give <item> to <NPC>, e.g., give potion to guard",
                    "give <item>, e.g., give sword",
                    "give <quantity> <item> to <npc>, e.g., give 5 potion to guard",
                ]
            },
            "fast travel to": {
                "description": "Instantly travel to a different world or major location.",
                "examples": ["fast travel to Shadowlands", "teleport to Mystic Peaks", "I want to fast travel to the Great Bazaar"],
                "parsing_guideline": "Identify the target world or major location for fast travel from the command.",
                "command_to_perform": "fast travel to",
                "command_format": "fast travel to <world/location>, e.g., fast travel to Shadowlands"
            },
            "look around": {
                "description": "Describe the surroundings in the current location.",
                "examples": ["look around", "survey the area", "what do I see"],
                "parsing_guideline": "Interpret as a request to describe the player's current surroundings.",
                "command_to_perform": "look around",
                "command_format": "look around, e.g., look around"
            },
            "where am i": {
                "description": "Provide the player's current location and description.",
                "examples": ["where am I", "what is this place"],
                "parsing_guideline": "Recognize this as a request for current location information.",
                "command_to_perform": "where am i",
                "command_format": "where am i, e.g., where am i"
            },
            "help": {
                "description": "Display the help menu.",
                "examples": ["help", "I need help"],
                "parsing_guideline": "Interpret as a request for the help menu.",
                "command_to_perform": "help",
                "command_format": "help, e.g., help"
            }
        }

        # Extract relevant player and world data for context
        current_location = player_data.location.get("location/sublocation", "Unknown Location")
        inventory = player_data.inventory
        current_location_data = self.world_builder.find_location_data(current_location)

        # Add contextual information about the player and the world to the prompt
        player_context = f"Player's current location: '{current_location}'. Player's inventory: {inventory}."
        location_context = f"Current location data: {current_location_data}"

        # Construct the prompt with command context and examples
        prompt = (
            f"As an AI interpreting player commands in a role-playing game, consider the following command dictionary and the player's current situation:\n"
            + "\n".join([
                f"- {cmd}: {desc['description']} (e.g., {', '.join(desc['examples'])}). "
                f"Parsing guideline: {desc['parsing_guideline']}. "
                f"Command to perform: {desc.get('command_to_perform', 'N/A')}. "
                f"Command format: {desc.get('command_format', 'N/A')}"
                for cmd, desc in command_dictionary.items()
            ])
            + "\n\n"
            + player_context + "\n" + location_context + "\n\n"
            f"Player's command: '{command}'\n"
            "Please interpret this command by providing a structured response with 'Action', 'Subject', and 'Quantity' (if applicable). Format the interpretation exactly as shown in these examples:\n"
            "- 'Action: go to, Subject: Great Bazaar'\n"
            "- 'Action: give, Subject: health potion, Quantity: 3'\n\n"
            "Ensure the response follows this format strictly for ease of parsing."
        )
        return prompt
    
    def determine_action_from_interpretation(self, interpreted_command):
        print(f"Determining action from interpreted command: {interpreted_command}")
        command, details, quantity = self.parse_interpreted_command(interpreted_command)
        print(f"Extracted command: {command}, details: {details}, quantity: {quantity}")

        if command == "move to":
            return self.world_builder.move_player(details)

        elif command == "talk to":
            # Ensure that only the NPC name is passed to handle_talk_to
            npc_name = details.strip("'")  # Strip any extraneous characters like quotes
            return self.world_builder.handle_talk_to(command, npc_name)


        elif command in ["take", "give"]:
            details = details.strip("'")
            return self.world_builder.handle_give_take(command, details)

        elif command == "examine":
            details = details.strip("'")
            return self.world_builder.examine_item(details)

        elif command in ["open"]:
            details = details.strip("'") 
            return self.world_builder.handle_open(details)
        
        elif command in ["close"]:
            details = details.strip("'")
            return self.world_builder.handle_close(details)

        elif command == "fast travel to":
            # Strip any extraneous characters like quotes from details
            ic('fast travel command detected')
            world_name = details.strip("'")
            ic(f'Command: {command}, World name: {world_name}')
            return self.world_builder.fast_travel_to_world(world_name)


        elif command in ["look around", "where am i", "help"]:
            return self.world_builder.simple_command_handler(command)

        else:
            print(f"Unknown command received: {command}")
            return self.world_builder.display_help()

    def parse_interpreted_command(self, interpreted_command):
        print(f"Extracting command details from: {interpreted_command}")
        
        # Regular expressions to extract Action, Subject, and Quantity
        # Stops at a comma or a newline
        action_match = re.search(r"Action:\s*([^\n,]+)", interpreted_command)
        subject_match = re.search(r"Subject:\s*([^\n,]+)", interpreted_command)
        quantity_match = re.search(r"Quantity:\s*(\d+)", interpreted_command)

        # Initialize default values
        action = "unknown"
        subject = ""
        quantity = 1  # Default quantity

        # Extract Action
        if action_match:
            action = action_match.group(1).lower().strip()

        # Extract Subject
        if subject_match:
            subject = subject_match.group(1).strip()

        # Extract Quantity
        if quantity_match:
            quantity = int(quantity_match.group(1))

        print(f"Extracted Action: {action}, Subject: {subject}, Quantity: {quantity}")
        return action, subject, quantity

        
    def generate_narrative_response(self, action_response, original_command):
        print(f"Generating narrative response for action: {action_response}, original command: {original_command}")
        player_data = self.player_sheet
        current_location = player_data.location.get("location/sublocation", "Unknown Location")
        location_data = self.world_builder.find_location_data(current_location)
        print(f"Current location data for narrative: {location_data}")

        prompt = self.construct_narrative_prompt(action_response, original_command, location_data, current_location)
        print(f"Narrative generation prompt: {prompt}")
        narrative_response = self.generate_ai_response(prompt)
        print(f"Generated narrative response: {narrative_response}")

        return narrative_response

    def construct_narrative_prompt(self, action_response, original_command, location_data, current_location):
        game_context = "'Odyssey' by NexaSphere is a virtual escape for many, including the player, until an unexpected event trapped users inside this digital realm. Now, they must unravel 'Odyssey's' mysteries to find a way back to reality."

        visible_items = [item for item in location_data.get('items', []) if item.get('visible', True)]
        visible_features = location_data.get('features', [])
        paths = location_data.get('paths', {})
        exits = [exit for exit in paths.values()]  
        containers = location_data.get('containers', [])

        # Filter rooms based on their 'visible' flag
        rooms = [room for room in location_data.get('rooms', []) if room.get('visible', True)]

        # Construct descriptions
        items_description = "Visible items: " + ", ".join([item['name'] for item in visible_items]) if visible_items else "No visible items."
        features_description = "Visible features: " + ", ".join(visible_features) if visible_features else "No visible features."
        paths_description = "Available paths: " + ", ".join(paths.keys()) if paths else "No visible paths."
        exits_description = "Exits: " + ", ".join(exits) if exits else "No visible exits."
        containers_description = "Containers: " + ", ".join([container['name'] for container in containers]) if containers else "No visible containers."
        rooms_description = "Rooms: " + "; ".join([f"{room['name']}: {room['description']}" for room in rooms]) if rooms else "No visible rooms."

        # Construct NPC descriptions
        npcs = location_data.get('npcs', [])
        npc_descriptions = "NPCs: " + "; ".join([f"{npc['name']}: {npc['description']}" for npc in npcs]) if npcs else "No NPCs in sight."

        # HTML format example
        html_format_example = "<html><body></body></html>"

        # Construct the actual prompt
        prompt = (
            f"Based on the following information, and replying in html format, generate a narrative response for a role-playing game:\n"
            f"- Game context: {game_context}\n"
            f"- HTML format example: {html_format_example}\n"
            f"- Player's original command: '{original_command}'\n"
            f"- Action taken: {action_response}\n"
            f"- Current location: {current_location}\n"
            f"- Location description: {location_data.get('description', 'An unknown location.')}\n"
            f"- {items_description}\n"
            f"- {features_description}\n"
            f"- {containers_description}\n"
            f"- {paths_description}\n"
            f"- {rooms_description}\n"
            f"- {npc_descriptions}\n\n"
            "Narrative Response:"
        )
        return prompt