import os
import dotenv
dotenv.load_dotenv()
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

model_engine = "text-davinci-003"
prompt = "Pretend you are an artificial intelligence running inside a cozmo robot. I will chat with you."

# Generate a response
def ask(prompt):
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        top_p=1,
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].text

response = ask(prompt)
print(response)

# TODO: should append context of conversation
while 1:
    txt = input("Human:")
    print(ask(txt))