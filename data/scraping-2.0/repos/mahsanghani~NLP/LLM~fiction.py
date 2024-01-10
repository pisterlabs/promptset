import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="List 10 science fiction books:",
  temperature=0.5,
  max_tokens=200,
  top_p=1.0,
  frequency_penalty=0.52,
  presence_penalty=0.5,
  stop=["11."]
)

# Prompt
# List 10 science fiction books:
# Sample response
# 1. 1984 by George Orwell
# 2. The War of the Worlds by H.G. Wells
# 3. Dune by Frank Herbert
# 4. Frankenstein by Mary Shelley
# 5. Ender's Game by Orson Scott Card
# 6. The Hitchhiker's Guide to the Galaxy by Douglas Adams
# 7. The Martian Chronicles by Ray Bradbury
# 8. Brave New World by Aldous Huxley 
# 9. Do Androids Dream of Electric Sheep? By Philip K Dick 
# 10. I, Robot by Isaac Asimov