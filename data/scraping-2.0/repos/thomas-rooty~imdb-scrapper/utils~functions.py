import openai
import streamlit as st


class Processor:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = st.secrets["OPENAI_API_KEY"]

    def translate_text(self, text):
        response = self.openai.Completion.create(
            engine="davinci",
            prompt=f"Translate from English to French:\nEnglish: {text}\nFrench:",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return response.choices[0].text

    def summarize_text(self, text):
        response = self.openai.Completion.create(
            engine="davinci",
            prompt=f"Original text: {text}\nSummary: ",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return response.choices[0].text

    def process_command(self, command):
        if command.startswith('/summarize'):
            text_to_summarize = command[len('/summarize '):]
            print(text_to_summarize)
            return self.summarize_text(text_to_summarize)
        elif command.startswith('/translate'):
            text_to_translate = command[len('/translate '):]
            print(text_to_translate)
            return self.translate_text(text_to_translate)
        else:
            return "Unknown command"
