import openai
import dotenv
import os
dotenv.load_dotenv()

class gpt:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.prompt = '''The following is a conversation with an AI assistant. The assistant is empathetic, patient and professional.
        Its approach is similar to the UK charity Samaritans. The AI never offers to make a phone call for the human. 
        AI: Would you like to start talking about what's on your mind? 
        Human: '''
        self.start_sequence = "\nAI: "
        self.restart_sequence = "\nHuman: "

    def get_response(self,msg):
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=self.prompt + msg + self.start_sequence,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
        )
        return response
