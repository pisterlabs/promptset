import openai
import pandas as pd
import os
from dotenv import load_dotenv
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt


class PromptingGPT:

    # Load API key and organization from environment variables
    load_dotenv("secrets.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    ClientOpenAi = openai.OpenAI(
            api_key= openai.api_key,
            organization= openai.organization
        )

    conversation_history = []

    def make_prompts(self, prompt):


        # Combine previous messages with the current prompt
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for msg in self.conversation_history:
            messages.append({'role': 'user', 'content': msg})

        messages.append({'role': 'user', 'content': prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )

        # Extract and print the model's reply
        reply = response.choices[0].message.content
        print(reply)

        # Update conversation history
        self.conversation_history.append(prompt)
        self.conversation_history.append(reply)

    def chat_prompts(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        state = True
        while state:
            pat_close = re.compile(
                r'(Bye|goodnight|ok thank you)', flags=re.IGNORECASE)
            message = input("You: ")
            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                chat_completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
            answer = chat_completion.choices[0].message.content
            print(f"ChatGPT: {answer}")
            messages.append({"role": "assistant", "content": answer})
            if re.search(pat_close, message):
                state = False


