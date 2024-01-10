import openai


class NameJokeChain:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.memory = []

    def add_to_memory(self, message):
        self.memory.append(message)
        if len(self.memory) > 3:
            self.memory = self.memory[-3:]

    def generate_response(self, message):
        self.add_to_memory(message)

        prompt = self.construct_prompt()
        response = self.generate(prompt)

        # return response.choices[0].text.strip()
        return response['choices']

    def construct_prompt(self):
        prompt = ""
        for msg in self.memory:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f"{content}\n"
            elif role == 'user':
                prompt += f"You said: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant said: {content}\n"
        return prompt

    def generate(self, prompt):
        openai.api_key = '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'
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )


chain = NameJokeChain()

user_message = {
    "role": "user",
    "content": "Hello, my name is John"
}

response = chain.generate_response(user_message)
print(response)
