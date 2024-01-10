import openai

class GenAIForge:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_generated_response(self, prompt):
        """_summary_
        Args:
            prompt (_type_): _description_
            api_key (_type_): _description_
        """
        #OpenAI API key
        openai.api_key = self.api_key

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )

        return response["choices"][0]["message"]["content"]
