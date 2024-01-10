import os

import openai

openai.api_key = "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"
#os.getenv("OPENAI_API_KEY")

import csv
from pprint import pprint


csv_filename = 'newsample.csv'

all_prods = []
cleaned_prods =[]

with open(csv_filename) as f:
    reader = csv.DictReader(f)

    for row in reader:
        all_prods.append(row)
        # print(row)
temp_prods = [x for x in all_prods if x["Variant Price"] != ""]
pprint(temp_prods)
    

# conversation_history = """
# Marv is a friendly chatbot that answers questions with cheerful responses:

# You: How many pounds are in a kilogram?
# Marv: There are 2.2 pounds in a kilogram, let me know if you need more help.
# You: What does HTML stand for?
# Marv: A search from google says it stands for hypertext markup language, hope this solves your query.
# You: When did the first airplane fly?
# Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights.
# You: What is the meaning of life?
# Marv: I’m not sure. I’ll ask my friend Google.
# You: 
# """

# def generatePrompt(userInput):
#     global conversation_history
#     conversation_history = conversation_history + userInput.strip() + "\nMarv: "
#     return conversation_history

# def generateResponse(userInput):
#     global conversation_history
#     thisPrompt = generatePrompt(userInput)
#     response = openai.Completion.create(
#         model="text-davinci-002",
#         max_tokens=4097 - len(thisPrompt),
#         prompt=thisPrompt,
#         temperature=0.6
#     )
#     reply = response["choices"][0]["text"].strip()
#     conversation_history = conversation_history + reply + "\nYou: "
#     # print(" !> debug:  ", )
#     return reply

# while(True):
#     userTxt = input(" ?> ")
#     if userTxt == "break":
#         break
#     print(" !> ", generateResponse(userTxt))

# print("++++++++ CONVERSATION HISTORY ++++++++")
# print(conversation_history)