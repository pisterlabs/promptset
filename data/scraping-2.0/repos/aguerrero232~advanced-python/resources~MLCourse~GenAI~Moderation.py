import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

moderation = openai.Moderation.create(
  input="Kill 'em all!",
)

print(moderation.results)



