import openai
class GPTtemplate:
    def __init__(self,API_KEY):
        openai.api_key = API_KEY
        self.goals = {} # goal is a dictionary of {goal prompt:finish percent}
        
