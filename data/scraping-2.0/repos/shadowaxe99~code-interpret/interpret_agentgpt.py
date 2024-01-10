import openai

class AgentGPT:
    def __init__(self):
        self.model = 'text-davinci-003'

    def get_response(self, prompt):
        response = openai.Completion.create(engine=self.model, prompt=prompt, max_tokens=150)
        return response