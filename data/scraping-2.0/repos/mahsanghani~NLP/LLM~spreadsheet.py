import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="A two-column spreadsheet of top science fiction movies and the year of release:\n\nTitle |  Year of release",
  temperature=0.5,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# A two-column spreadsheet of top science fiction movies and the year of release:

# Title |  Year of release
# Sample response
# Alien | 1979 
# Blade Runner | 1982 
# The Terminator | 1984 
# The Matrix | 1999 
# Avatar | 2009 
# Interstellar | 2014 
# Ex Machina | 2015 
# Arrival | 2016 
# Ready Player One | 2018