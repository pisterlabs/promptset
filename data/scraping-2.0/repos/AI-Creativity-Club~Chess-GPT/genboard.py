#https://platform.openai.com/docs/api-reference/chat/create
import openai

#Attempt at creating a board using the openai module, prompt needs work ...

openai.api_key = ("API_KEY")
openai.Model = ("text-davinci-003")

completion = openai.Completion.create(
        model="text-davinci-003",
        prompt="Create a csv file that represents a chess board made up of 0s and 1s, add a random pattern of x's but retaining the chess board", #Insert a prompt for the openai davinci
        max_tokens=140,
        temperature=0.3
    )
print(completion.choices[0].text)