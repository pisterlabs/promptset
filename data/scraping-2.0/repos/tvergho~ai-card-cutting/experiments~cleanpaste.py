import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

f = open("sample.txt", "r")
text = f.read()
f.close()

# Slice the text into 4000 character chunks
chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]

for i in range(min(3, len(chunks))):
  chunk = chunks[i]
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Sanitize the following PDF input by removing page numbers, repeated headers and footers, contact information, footnotes, extra space between lines, and any and all extraneous information. Preserve paragraph integrity. Otherwise, do not alter a single word of the original text.\n\n###\n\nPDF Input: " + chunk + "\n\n###\n\Output:",
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
  )
  prediction = response["choices"][0]["text"]
  # print("Chunk " + str(i) + ":")
  print(prediction)