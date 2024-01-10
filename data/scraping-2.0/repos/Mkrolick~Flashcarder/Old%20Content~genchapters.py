import os
import openai
import dotenv
import prompts

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

book_name = input("Enter the name of the book: ")

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"List out all of the chapters of the book: {book_name} in a python list format. In the format [ChapterOneName, ChapterTwoName, ...]",
    temperature=0.7,
    max_tokens=709,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# save out put to a text file
with open('output.txt', 'w') as f:
    f.write(response['choices'][0]['text'])
