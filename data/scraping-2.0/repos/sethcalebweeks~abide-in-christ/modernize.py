import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """
Translate the following paragraph into modern English that is simple to read while preserving the meaning and relative length of the original text. Do not add or take away anything from the original text. Write in the style of Andrew Murray if he was living in the year 2020. Replace all quoted Bible verses with ESV.

Paragraph:
{}

Translation:
"""


def modernize(paragraph):
  response = openai.Completion.create(
    model = "text-davinci-003",
    prompt = prompt.format(paragraph),
    max_tokens = 3000,
    temperature = 0.4
  )
  return response.choices[0].text.strip()


for line in open("original/chapter-31.txt"):
  modernized = modernize(line)
  open("book/chapter-31-the-glorified-one.md", "a").write(f"{modernized}\n\n")

