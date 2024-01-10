import openai
import json

MODELS = ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-002', 'text-davinci-003', 'davinci-002', 'babbage-002', 'gpt-3.5-turbo', 'gpt-4']
ROLES = ["system", "user", "assistant"]

class ChatCompletionAgent:
    def __init__(self, api_token, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1024, n=1, stop=None, presence_penalty=0, frequency_penalty=0, messages = [], behavior:str = "You are a helpful assistant."):
        self.api_token = api_token
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.behavior = behavior
        self.messages = messages
        self.on_init()

    def on_init(self):
        if self.behavior != None and self.messages == []:
            self.add_message(0, self.behavior)
    
    def set_param(self, model=None, temperature=None, max_tokens=None, n=None, stop=None, presence_penalty=None, frequency_penalty=None, best_of=None, behavior=None):
        # Mettez à jour les paramètres de l'agent avec de nouvelles valeurs
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if n is not None:
            self.n = n
        if stop is not None:
            self.stop = stop
        if presence_penalty is not None:
            self.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty
        if behavior is not None:
            self.behavior = behavior

    def add_message(self, role:int, content:str):
        self.messages.append({"role": ROLES[role], "content": content})

    def delete_message(self):
        self.messages = []

    def _get_text_thread(self):

        openai.api_key = self.api_token

        response = openai.ChatCompletion.create(
            model=self.model,
            messages = self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        
        return response
    
    def get_text(self, prompt):
        self.add_message(1, prompt)
        print(self.messages)
        response = self._get_text_thread()
        print(response)
        new_message = response['choices'][0]['message']['content']
        self.add_message(2, new_message)
        print(self.messages)
        return new_message
    
    def save_results_to_file(self, results, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(results, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'enregistrement des résultats : {e}")