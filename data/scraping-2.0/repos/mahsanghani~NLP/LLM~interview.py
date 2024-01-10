import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Create a list of 8 questions for my interview with a science fiction author:",
  temperature=0.5,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Create a list of 8 questions for my interview with a science fiction author:
# Sample response
# 1. What inspired you to write science fiction?
# 2. What themes do you typically explore in your work?
# 3. How has the genre of science fiction evolved over time?
# 4. What do you think sets your work apart from other science fiction authors?
# 5. What advice would you give to aspiring science fiction authors?
# 6. What are some of the most challenging aspects of writing science fiction?
# 7. How has your writing process changed over the years?
# 8. What do you think is the most important thing for readers to take away from your work?