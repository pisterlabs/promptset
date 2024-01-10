from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

text = input("Enter text to moderate: ")

response = client.moderations.create(input=text)

output = response.results[0]

print(output)
