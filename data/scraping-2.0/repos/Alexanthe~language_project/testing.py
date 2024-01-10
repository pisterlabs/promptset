import requests
from datetime import date, timedelta
import api_key
import os
import openai
import beeai

# from wordDictExample import *


openai.api_key = api_key.openai_api_key

word = "Obreptitious"
beeai.generateHive(word)

def generate_prompt(word, number, style):
    return """Write a {sty} word {num} story for and with the word '{wor}'""".format( wor=word.capitalize(), num=number,sty=style)

number=10
style="alex"
print(generate_prompt(word, number, style))
# response = openai.Completion.create(
#             model="text-davinci-003",
#             prompt=generate_prompt(word),
#             max_tokens=200,
#             temperature=0.6,
#         )
# print(response)
# print(response.choices[0].text)


# today = date.today()

# todayWordNik = today - timedelta(days=1)

# query = {"date": todayWordNik, "api_key": api_key.api_key}

# response = requests.get(
#     "https://api.wordnik.com/v4/words.json/wordOfTheDay", params=query
# )

# wordDict = response.json()

# ## Get Word ####
# word = wordDict["word"]

# print(word)

# ## Get Definitions ####
# definitions = []
# for each in wordDict["definitions"]:
#     definitions.append(each)

# # Function to get the different sources and texts definitions (Might have multiple definitions)
# def getSpecificDef(definitions):
#     source = [] 
#     text = []

#     for each in definitions:
#         source.append(each["source"])
#         text.append(each["text"])
    
#     return source,text

# source,text = getSpecificDef(definitions)



# ## Get Examples ####
# examples = []
# for each in wordDict["examples"]:
#     examples.append(each)

# print(len(examples)) # Cam create a py function to just get specific dictionary key values such as "text"

# ## Get Published Dates ####
# publishedDate = wordDict["pdd"]

# ## Get Note ####
# note = wordDict["note"]

###########################################################################

## Using a different query api type ####
# query = {
#     "hasDictionaryDef": True,
#     "api_key": api_key.api_key,
#     "includePartOfSpeech": True,
#     "excludePartOfSpeech": True,
#     "maxCorpusCount": -1,
#     "minDictionaryCount": 1,
#     "maxDictionaryCount": -1,
#     "minLength": 5,
#     "maxLength": -1
# }

# response = requests.get(
#     "https://api.wordnik.com/v4/words.json/randomWord", params=query
# )

# wordsDict = response.json()

# # wordsDict = {"id": 0, "word": "corner"}  # Example

# word = wordsDict["word"]

# print(word)

# query = {"useCanonical": True, "api_key": api_key.api_key}

# response = requests.get(
#     "https://api.wordnik.com/v4/word.json/" + word + "/topExample", params=query
# )

# wordExample = response.json()

# print(wordExample)


# query = {'lat':'45', 'lon':'180'}
# response = requests.get('http://api.open-notify.org/iss-pass.json', params=query)
# print(response.json())
