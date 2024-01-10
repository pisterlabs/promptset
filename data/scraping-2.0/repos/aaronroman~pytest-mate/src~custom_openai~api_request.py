import openai


# Clase OpenAIModel
class OpenAIModel:
    def __init__(self, model_name: str, system_prompt: str, api_key: str):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.api_key = api_key

    # List all available models
    @staticmethod
    def list_models() -> list:
        return openai.Model.list()

    # Request to OpenAI
    def generate_response(self, prompt: str) -> str:
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
        )
        return response["choices"][0]["message"]["content"]
