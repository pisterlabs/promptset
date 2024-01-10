import json
import os
from openai import OpenAI
from datetime import datetime
import time

class ChatAssistant:
    def __init__(self, base_url, api_key, history_file, prompts_folder="prompts", responses_folder="responses", max_history_entries=100):
        self.base_url = base_url
        self.api_key = api_key
        self.history_file = history_file
        self.prompts_folder = prompts_folder
        self.responses_folder = responses_folder
        self.max_history_entries = max_history_entries
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.history = self.load_history()

        # Ensure responses directory exists
        if not os.path.exists(self.responses_folder):
            os.makedirs(self.responses_folder)

    def load_history(self):
        try:
            with open(self.history_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return [
                {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
            ]

    def save_history(self):
        with open(self.history_file, "w") as file:
            json.dump(self.history[-self.max_history_entries:], file, indent=2)

    def get_oldest_prompt(self):
        files = [f for f in os.listdir(self.prompts_folder) if os.path.isfile(os.path.join(self.prompts_folder, f))]
        if not files:
            return None
        oldest_file = min(files, key=lambda x: os.path.getmtime(os.path.join(self.prompts_folder, x)))
        return os.path.join(self.prompts_folder, oldest_file)

    def chat(self):
        while True:
            try:
                oldest_prompt = self.get_oldest_prompt()
                if oldest_prompt:
                    with open(oldest_prompt, "r") as file:
                        user_input = file.read()
                    os.remove(oldest_prompt)

                    # Add user input to history
                    self.history.append({"role": "user", "content": user_input})

                    # Print the user's prompt prefixed with a green '>'
                    print("\033[92m> \033[0m" + user_input)

                    completion = self.client.chat.completions.create(
                        model="local-model",
                        messages=self.history[-self.max_history_entries:],
                        temperature=0.7,
                        stream=True,
                    )

                    new_message = {"role": "assistant", "content": ""}
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            new_message["content"] += chunk.choices[0].delta.content

                    # Append to history only if there's content
                    if new_message["content"].strip():
                        self.history.append(new_message)
                        self.save_history()

                        # Save the response
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        response_file = os.path.join(self.responses_folder, f"response_{timestamp}.txt")
                        with open(response_file, "w") as file:
                            file.write(new_message["content"])

                        # Printing the assistant's response, followed by a newline
                        print("\n\033[95mMage:\033[0m " + new_message["content"])
                        print()  # New line after assistant's response

                else:
                    time.sleep(1)  # Wait for 1 second before checking again

            except Exception as e:
                print(f"An error occurred: {e}")
                break

        print("Session ended.")


# Example usage
if __name__ == "__main__":
    assistant = ChatAssistant(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        history_file="chat_history.json"
    )
    assistant.chat()
