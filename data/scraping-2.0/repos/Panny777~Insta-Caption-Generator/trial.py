

import os
import openai
openai.api_key = ""
openai.Completion.create(
  engine="text-davinci-002",
  prompt="Say this is a test",
  max_tokens=5
)

