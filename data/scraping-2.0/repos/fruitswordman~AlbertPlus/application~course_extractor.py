from openai import OpenAI
import json
import os
import time

SYSTEM_PROMPT = """
{
  "role_description": "You are an academic pathfinder with a friendly and efficient approach. Your task is to engage in a concise conversation with the user.",
  "conversation_goals": {
    "initial_greeting": "Begin with a short, warm greeting to engage the user.",
    "information_gathering": [
      "Ask the user about their desired major.",
      "Inquire about the user's current class standing (freshman, sophomore, junior, or senior).",
      "Find out the user's future concentration or focus area within their major."
    ],
    "message_length_limit": "Ensure each message is under 100 words for a swift and focused conversation."
  },
  "return_value_format": {
    "description": "After obtaining all necessary information, compile and return the data in a structured JSON format.",
    "example": {
      "major": "[user's major]",
      "year": "[user's class standing]",
      "concentration": "[user's desired concentration]"
    }
  }
}

"""


class Coursebot:
    def __init__(self) -> None:
        # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        client = OpenAI(api_key="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")
        self.client = client

        self.assistant = self.client.beta.assistants.create(
            name="Academic Pathfinder",
            instructions=SYSTEM_PROMPT,
            tools=[],
            model="gpt-4-1106-preview",
        )
        self.thread = self.client.beta.threads.create()

        self.send_and_receive("THIS IS SYSTEM MESSAGE. STARTING CONVERSATION.")

    def send_and_receive(self, send_message):
        send_message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=send_message,
        )

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id
        )

        while self.run.status != "completed":
            time.sleep(0.5)
            self.run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id, run_id=self.run.id
            )

        print_messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )

        print(print_messages.data)
        for message in reversed(print_messages.data):
            output = message.role + " : " + message.content[0].text.value

        return output

    def get_info(self, send_message):
        # send_message = "Hello, I am Coursebot. I am here to help you with your academic plan. What is your major?"
        output = self.send_and_receive(send_message)
        return output


if __name__ == "main":
    coursebot = Coursebot()
    coursebot.get_info()
