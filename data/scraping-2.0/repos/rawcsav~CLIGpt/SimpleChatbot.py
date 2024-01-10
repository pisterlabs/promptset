import asyncio
import configparser
import os

import openai
import tiktoken
from colorama import init, Fore

init(autoreset=True)


class ChatAssistant:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.load_config()

        self.encoding = tiktoken.encoding_for_model(self.model)
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            },
        ]

    def load_config(self):
        self.config.read("config.ini")

        # Check for API_KEY in config, if not found, ask the user
        if not self.config.has_option("Settings", "API_KEY"):
            openai.api_key = input("Enter your OpenAI API key: ")
            if not self.config.has_section("Settings"):
                self.config.add_section("Settings")
            self.config.set("Settings", "API_KEY", openai.api_key)
        else:
            openai.api_key = self.config.get("Settings", "API_KEY")

        # Check for MODEL in config, if not found, set to default 'gpt-3.5-turbo'
        if not self.config.has_option("Settings", "MODEL"):
            self.model = "gpt-3.5-turbo"
            self.config.set("Settings", "MODEL", self.model)
        else:
            self.model = self.config.get("Settings", "MODEL")

        # Load or set default values for other parameters
        self.temperature = float(self.config.get("Settings", "TEMPERATURE", fallback="0.5"))
        self.system_prompt = self.config.get("Settings", "SYSTEM_PROMPT", fallback="You are a helpful assistant.")

        # Save the updated configuration to the file
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)

        self.set_token_limits()

    def prompt_for_model(self):
        models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        ]
        print("Select a model:")
        for i, model in enumerate(models):
            print(f"{i}. {model}")
        choice = int(input("Enter choice number: "))
        return models[choice]

    def set_token_limits(self):
        self.max_tokens = (
            31000 if "gpt-4-32k" in self.model
            else 7000 if "gpt-4" in self.model
            else 15000 if "gpt-3.5-turbo-16k" in self.model
            else 4000
        )
        self.truncate_limit = (
            30500 if "gpt-4-32k" in self.model
            else 6500 if "gpt-4" in self.model
            else 14500 if "gpt-3.5-turbo-16k" in self.model
            else 3500
        )

    def set_openai_key(self):
        if os.path.exists("config.txt"):
            with open("config.txt", "r") as f:
                openai.api_key = f.read().strip()
        else:
            openai_key = input("Enter your OpenAI API key: ")
            openai.api_key = openai_key
            with open("config.txt", "w") as f:
                f.write(openai_key)

    def truncate_conversation(self):
        while True:
            if self.get_token_count() > self.truncate_limit and len(self.conversation_history) > 1:
                self.conversation_history.pop(1)
            else:
                break

    def get_token_count(self):
        num_tokens = 0
        for message in self.conversation_history:
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += 5
        num_tokens += 5
        return num_tokens

    async def monitor_user_input(self):
        await asyncio.get_event_loop().run_in_executor(None, input)

    async def chat_with_gpt4(self, prompt):
        self.truncate_conversation()
        full_response = ""

        interrupt_event = asyncio.Event()
        ai_task = asyncio.create_task(self.get_ai_response(prompt, full_response))
        monitor_task = asyncio.create_task(self.monitor_user_input())

        done, pending = await asyncio.wait([ai_task, monitor_task], return_when=asyncio.FIRST_COMPLETED)

        if monitor_task in done:
            ai_task.cancel()
            print(Fore.RED + "\nAI response interrupted by user!")

            if self.conversation_history[-1]['role'] == 'user':
                self.conversation_history.pop()
            return full_response

        monitor_task.cancel()
        return await ai_task

    async def get_ai_response(self, prompt, full_response):
        chat = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=self.conversation_history + [{"role": "user", "content": prompt}],
            temperature=self.temperature,  # Use the temperature from the config
            max_tokens=self.max_tokens,
            stream=True,
        )
        first_chunk = True
        async for chunk in chat:
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                full_response += content
                if first_chunk:
                    print(f"\n{Fore.CYAN}Assistant: " + content, end="")
                    first_chunk = False
                else:
                    print(f"{Fore.CYAN}" + content, end="")
        print("\n")
        return full_response

    async def main(self):
        print(Fore.YELLOW + "Starting Chat Assistant...")
        print(Fore.RED + "\nPress ENTER at any time to interrupt the AI response.")
        print(
            "This allows you to stop the AI's response if it's taking too long or if you wish to send another message.")
        while True:
            user_prompt = input(f"\n{Fore.GREEN}You: ")
            if user_prompt.lower() == "exit":
                break
            response_text = await self.chat_with_gpt4(user_prompt)
            self.conversation_history.extend([
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
            ])


if __name__ == "__main__":
    assistant = ChatAssistant()
    asyncio.run(assistant.main())
