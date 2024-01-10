import openai as ai
import json

ai.api_key = "YOUR_API_KEY_HERE"

# Import text prompt
with open("prompt.txt") as pro:
	prompt = pro.read()
	
# The Model
returns = ai.Completion.create(
    engine="davinci", # OpenAI has made four text completion engines available, named davinci, ada, babbage and curie. We are using davinci, which is the most capable of the four.
    prompt=prompt, # The text file we use as input (step 3)
    max_tokens=100, # how many maximum characters the text will consists of.
    temperature=0.9, # a number between 0 and 1 that determines how many creative risks the engine takes when generating text.,
    top_p=1, # an alternative way to control the originality and creativity of the generated text.
    n=1, # number of predictions to generate
    frequency_penalty=0.5, # a number between 0 and 1. The higher this value the model will make a bigger effort in not repeating itself.
    presence_penalty=0.9 # a number between 0 and 1. The higher this value the model will make a bigger effort in talking about new topics.
)


text = returns['choices'][0]['text']

print(text)
