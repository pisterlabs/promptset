

import os
import openai
openai.api_key = "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"
openai.Completion.create(
  engine="text-davinci-002",
  prompt="Say this is a test",
  max_tokens=5
)

