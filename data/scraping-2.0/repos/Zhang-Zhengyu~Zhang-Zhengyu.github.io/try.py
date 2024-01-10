import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I didn't like the new Batman movie!
Sentiment:
"""

response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt,
              max_tokens=100,
              temperature=0
            )

print(response)