import openai
import os

# Set up the OpenAI API client
openai.api_key = "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"
# Set up the model and prompt
model_engine = "text-davinci-003"

# Generate a response

cond = True
while cond:
    prompt = input("Hi, I'm r47dzt3ch AI. How can I help you? ")
    completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
    )      
    response = completion.choices[0].text
    print(response)
    print("\nAre you want to continue?\n(enter)-Yes\t\t(1)-No")
    key = input("enter: ")
    os.system('cls')
    if str(key) =='1' or str(key)=='no':
        cond=False
