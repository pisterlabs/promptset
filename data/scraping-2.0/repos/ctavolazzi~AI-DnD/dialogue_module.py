# dialogue_module.py
import openai

from module_interface import ModuleInterface

class DialogueModule(ModuleInterface):
    def initialize(self, config, game_state):
        # Initialize dialogue module with OpenAI API configuration
        self.api_key = config['api_key']
        self.game_state = game_state
        # Further configuration as needed

    def update(self, game_state):
        # Logic to decide when to generate dialogue
        pass

    def terminate(self, game_state):
        # Clean up if necessary
        pass

    def generate_dialogue(self, prompt):
        # Ensure you have the "openai" package installed and your API key set
        openai.api_key = self.api_key

        try:
            response = openai.Completion.create(
              engine="text-davinci-003",
              prompt=prompt,
              max_tokens=150  # You can adjust the max tokens as per your requirement
            )
            return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            # Handle API errors
            print(f"OpenAI API error: {e}")
            return ""
        except Exception as e:
            # Handle other errors
            print(f"An error occurred: {e}")
            return ""

    def process_dialogue(self, dialogue):
        # Logic to process the generated dialogue
        pass