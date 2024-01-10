# %%
!source setkey.sh

# %%
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.Completion.create(
  engine="davinci",
  prompt="Create an outline for an essay about Walt Disney and his contributions to animation:\n\nI: Introduction",
  temperature=0.7,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# %%
print(response)


# %%
