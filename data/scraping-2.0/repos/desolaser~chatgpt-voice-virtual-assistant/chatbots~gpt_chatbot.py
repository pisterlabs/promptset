import openai
from decouple import config
from interfaces.chatbot import ChatbotInterface

OPENAI_API_KEY = config('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

class GPTChatbot(ChatbotInterface):
    def __init__(self, personality, character_name, verbose=False) -> None:
        self.chat_log = personality
        self.character_name = character_name
        self.verbose = verbose

    def get_response(self, text):
        self.start_sequence = "\n" + self.character_name
        self.restart_sequence = "\n\nPerson"

        if self.verbose:
            print("START OF CHAT LOG ===================================")
            print(self.chat_log)
            print("END OF CHAT LOG =====================================")

        prompt_text = f"{self.chat_log}{self.restart_sequence}: {text}{self.start_sequence}:"
        if self.verbose:
            print("START OF PROMPT ===================================")
            print(prompt_text)
            print("END OF PROMPT =====================================")

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_text,
            temperature=0.8,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.3,
            stop=[" Person: ", self.character_name + ": "]
        )
        answer = response['choices'][0]['text']
        self.chat_log += f"{self.restart_sequence}: {text}{self.start_sequence}:{answer}"

        return answer