import openai
import time

class OpenAIClient:
    def __init__(self, instructions, openai_api_key):
        self.client = openai.OpenAI(api_key=openai_api_key)

        file = self.client.files.create(
            file = open('store/messages.txt', 'rb'),
            purpose = 'assistants',
        )

        self.assistant = self.client.beta.assistants.create(
            name="Funny footbal show host",
            instructions = instructions,
            tools=[{"type": "retrieval"}],
            model="gpt-4-1106-preview",
            file_ids = [file.id],
        )

        self.thread = self.client.beta.threads.create()

    def send_user_input(self, user_input: str):
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_input
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

        i = 0
        while not run.status == 'completed':
            print("ChatGPDanny: Aan het typen" + '.' * i + '  ' + '\r', end='')
            time.sleep(0.2)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            i += 1
            if i > 3:
                i = 0


    def get_assistant_response(self):

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )

        response = messages.data[0].content[0].text.value

        return response
    

class DummyClient:
    def __init__(self, instructions, openai_api_key):
        pass

    def send_user_input(self, user_input: str):

        i = 0

        while i < 3:
            print("ChatGPDanny: Aan het typen" + '.' * i + '  ' + '\r', end='')
            time.sleep(0.3)
            i += 1

    def get_assistant_response(self):
        return "This is a response"