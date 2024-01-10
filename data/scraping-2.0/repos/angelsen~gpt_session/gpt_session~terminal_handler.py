import subprocess
from .openai_client import OpenAIClient

class TerminalSessionHandler:
    def __init__(self):
        self.client = OpenAIClient()
        self.last_command = None
        self.last_output = None
        self.conversation = []

    def start_session(self):
        while True:
            user_input = input("$ ")
            if user_input.startswith("???"):
                # Strip '???' and send the remaining string to the OpenAI API Client
                question = user_input[3:].strip()
                response = self.client.ask(question, self.last_command, self.last_output, self.conversation)
                print(response)
            elif user_input.startswith("??"):
                # Strip '??' and send the remaining string to the OpenAI API Client
                question = user_input[2:].strip()
                response = self.client.ask(question, conversation=self.conversation)
                print(response)
            else:
                # Execute the command as usual
                process = subprocess.Popen(user_input, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                print(stdout.decode())
                print(stderr.decode())
                # Update last_command and last_output
                self.last_command = user_input
                self.last_output = stdout.decode()

def main():
    handler = TerminalSessionHandler()
    handler.start_session()

if __name__ == "__main__":
    handler = TerminalSessionHandler()
    handler.start_session()
