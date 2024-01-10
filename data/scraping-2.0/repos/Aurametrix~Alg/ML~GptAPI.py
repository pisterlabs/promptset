################ misc

import openai
openai.api_key = open("/Users/simon/.openai-api-key.txt").read().strip()

for chunk in openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "Generate a list of 20 great names for sentient cheesecakes that teach SQL"
    }],
    stream=True,
):
    content = chunk["choices"][0].get("delta", {}).get("content")
    if content is not None:
        print(content, end='')


# This code is Apache 2 licensed:
# https://www.apache.org/licenses/LICENSE-2.0
import openai

class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content
      
      #####################
      ######  Hugging Face 
      ### GPT 2.0
      
      from transformers import pipeline

# Load the GPT-2 model
nlp = pipeline("text-generation", model="gpt2")

# Read the input text from a file
with open("input.txt", "r") as f:
    text = f.read()

# Generate text using the GPT-2 model
output = nlp(text, max_length=100)

# Print the generated text
print(output["generated_text"])
