import openai
import time

openai.api_key = input("Enter your OpenAI API key: ")

class QAAssistant:
    def __init__(self, file_path):
        self.client = openai.OpenAI()
        self.file = self._upload_file(file_path)
        self.assistant = self._create_assistant()
        self.thread = self._create_thread()

    def _upload_file(self, file_path):
        file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose='assistants'
        )
        return file

    def _create_assistant(self):
        return self.client.beta.assistants.create(
            name="Q/A Assistant",
            instructions="You are a Q/A chatbot, answering questions based on the uploaded file to provide the best response to the user.",
            model="gpt-4-1106-preview",
            tools=[{"type": "retrieval",
                    "config": {
                        "timeout": 10,
                        "max_memory": "512MB"
                    }}],
            description="A chatbot that answers questions based on the uploaded file.",
            file_ids=[self.file.id]
        )

    def _create_thread(self):
        return self.client.beta.threads.create()

    def ask_question(self, prompt):
        return input(prompt)

    def run_assistant(self):
        keep_asking = True
        while keep_asking:
            user_question = self.ask_question("\nWhat is your question? ")

            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_question
            )

            run = self.client.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant.id)

            run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
            while run_status.status != "completed":
                time.sleep(2)
                run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)

            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            last_message_for_run = None
            for message in reversed(messages.data):
                if message.role == "assistant" and message.run_id == run.id:
                    last_message_for_run = message
                    break

            if last_message_for_run:
                print(f"{last_message_for_run.content[0].text.value} \n")

            continue_asking = self.ask_question("Do you want to ask another question? (yes/no) ")
            keep_asking = continue_asking.lower() == "yes"

        print("Thank you for using the Q/A Assistant. Have a great day!")