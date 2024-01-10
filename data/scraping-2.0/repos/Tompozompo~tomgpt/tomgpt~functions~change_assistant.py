import os
from openai import OpenAI
from tomgpt.functions.chatfunction import ChatFunction
import tomgpt.flaskapp

class ChangeAssistantFunction(ChatFunction):
    @property
    def name(self):
        return "change_assistant"

    @property
    def description(self):
        return "Creates a new assistant and returns the new assistant ID."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "system_prompt": {
                    "type": "string",
                    "description": "The system prompt for the new assistant.",
                },
                "name": {
                    "type": "string",
                    "description": "The name for the new assistant.",
                },
                "model": {
                    "type": "string",
                    "description": "The openAI model to use.",
                },

            },
            "required": ["system_prompt, name"],
        }  
    def execute(self, **kwargs):
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        system_prompt = kwargs.get('system_prompt', 'Default system prompt')
        name = kwargs.get('name', 'Default name')
        model = kwargs.get('model', 'gpt-4')
        try:
            assistant = client.beta.assistants.create(
                name=name,  
                instructions=system_prompt,
                model=model,
            )
            flaskapp.ASSISTANT_ID = assistant.id
            return {"new_assistant_id": assistant.id}
        except Exception as e:
            return {"error": str(e)}