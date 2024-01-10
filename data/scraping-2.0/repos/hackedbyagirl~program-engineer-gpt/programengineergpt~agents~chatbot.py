import chromadb
from chromadb.config import Settings
import datetime
import json
import openai
import time

from time import sleep

from programengineergpt.utils.colors import Color
from programengineergpt.utils.display import Display
from programengineergpt.prompts.build_user_prompt import build_user_prompt
from programengineergpt.prompts.chat import SYSTEM_CHAT_PROMPT


class ChatBot:
    def __init__(self, code_collection):
        self.code_collection = code_collection
        self.openai_api = openai.ChatCompletion()
        self.chat_history = []
        self.model = "gpt-3.5-turbo-16k"
        self.temperature = 0.1
        self.chat_file = f"chat_log_{time.time()}.json"

    def launch(self):
        # Display banner
        Display.display_interactive_chat_banner()

        
        # Set initial system prompt
        self.chat_history.append({"role": "system", "content": SYSTEM_CHAT_PROMPT})
       
        # Get First Question and context
        Color.print("\n{G}Question: ")
        question = input()
        context = self.retrieve_context(question)

        # Build first prompt
        user_prompt = build_user_prompt(question, context)
        self.chat_history.append(user_prompt)

        # Launch first question
        self.next_step(self.chat_history)

    def interact(self):
        """
        Queries an index to allow conversational QA
        """
        # Launch initial ai interaction
        self.launch()

        # Engage in Interactive chat loop
        while True:
            # Get question
            Color.print("\n{G}Question: ")
            question = input()
            if question.lower() == "exit":
                self.save_chat_log()
                break

            # Build user prompt
            context = self.retrieve_context(question)
            user_prompt = build_user_prompt(question, context)
            self.chat_history.append(user_prompt)

            # continue chat
            self.next_step(self.chat_history)

    def retrieve_context(self, query):
        results = self.code_collection.query(
            query_texts=[query], include=["documents", "metadatas"], n_results=3
        )
        if results:
            return [
                (doc, meta["filename"])
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        else:
            return None

    def next_step(self, messages):
        max_retry = 7
        retry = 0
        while True:
            try:
                response = self.openai_api.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=self.temperature,
                )

                Color.print("\n{B}Answer: \n")
                chat = []
                for chunk in response:
                    delta = chunk["choices"][0]["delta"]
                    msg = delta.get("content", "")
                    print(msg, end="")
                    chat.append(msg)
                print()

                chat_response = {"role": "assistant", "content": "".join(chat)}          
                self.chat_history.append(chat_response)
                
                break

            except Exception as oops:
                print(f'\n\nError communicating with OpenAI: "{oops}"')
                if "maximum context length" in str(oops):
                    self.chat_history.pop(0)
                    print("\n\n DEBUG: Trimming oldest message")
                    continue
                retry += 1
                if retry >= max_retry:
                    print(f"\n\nExiting due to excessive errors in API: {oops}")
                    exit(1)
                print(f"\n\nRetrying in {2 ** (retry - 1) * 5} seconds...")
                sleep(2 ** (retry - 1) * 5)

    def save_chat_log(self):
        with open(self.chat_file, 'w') as f:
            json.dump(self.chat_history, f)