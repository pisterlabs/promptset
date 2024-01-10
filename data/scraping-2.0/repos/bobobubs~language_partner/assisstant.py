from openai import OpenAI
from prompts import init_prompt
from credentials import secret
import time

class LanguagePartnerChatbot:
    def __init__(self, api_key, init_prompt, model="gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.init_prompt = init_prompt
        self.model = model
        self.assistant = self.create_assistant()
        self.thread = self.create_thread()

    def create_assistant(self):
        return self.client.beta.assistants.create(
            name="Language Partner",
            instructions=self.init_prompt,
            model=self.model,
        )

    def create_thread(self):
        return self.client.beta.threads.create()

    def send_message(self, user_input):
        return self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_input,
        )

    def get_response(self):
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=self.init_prompt
        )

        while run.status in ["queued", "in_progress"]:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        for msg in messages.data:
            if msg.role == 'assistant':
                return msg.content[0].text.value

    def start_chat(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting chat...")
                break

            self.send_message(user_input)
            response = self.get_response()
            print("Assistant:", response)

if __name__ == "__main__":
    chatbot = LanguagePartnerChatbot(api_key=secret, init_prompt=init_prompt)
    chatbot.start_chat()
