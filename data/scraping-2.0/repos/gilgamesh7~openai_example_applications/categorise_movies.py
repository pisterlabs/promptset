import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

movie_to_categorise = input("Enter name of movie for categorisation  : ")

categorisation_prompt = f"Enter name of movie for categorisation as Action , Thriller , Drama , Horror , Western , Science Fiction , Drama , Crime , Comedy , Romance , Adventure, Slasher :\n\n{movie_to_categorise}\n\n1.",

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=categorisation_prompt,
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(f'1.{response["choices"][0]["text"]}')
