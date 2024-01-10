import openai
import json

class exploration():
    def __init__(self, conversation_file_name) -> None:
        openai.api_key = self._get_api_key("config.json")
        self.prompt = [{"role": "system", "content": "Are you ready?"},]
        self.conversation_file = conversation_file_name
        
    def chatbot(self):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=self.prompt
        )
        return response["choices"][0]["message"]
    
    def save_conversation(self):
        with open(self.conversation_file, 'w') as f:
            f.write(json.dumps(self.prompt))

    def _get_api_key(self, config_file):
        print("Initializing ChatGPT...")
        with open(config_file, "r") as f:
            config = json.load(f)
        return config["api_key"]

