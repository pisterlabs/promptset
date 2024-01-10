import openai
import streamlit as st


class GPT_API:
    def __init__(self):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        self.memory = True
        self.messages = []

    def system(self, prompt):
        self.messages.append({"role": "system", "content": prompt})

    def assistant(self, prompt):
        self.messages.append({"role": "assistant", "content": prompt})

    def chat(self, prompt, temperature=1, model="gpt-3.5-turbo"):
        self.messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            # temperature=temperature  # 0-2, degree of randomness
            # docs: https://platform.openai.com/docs/api-reference/chat
        )

        if self.memory:
            self.messages.append(
                {"role": "assistant", "content": response["choices"][0]["message"].content})
        else:
            self.messages = self.messages[:-1]
        return response.choices[0].message.content
