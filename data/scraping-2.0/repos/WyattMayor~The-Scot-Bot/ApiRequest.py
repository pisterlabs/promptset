#import necassary libraries
import time
import os
import openai

# Used to get a answer based on a given passage that was produced by PassageRanker.py
def RequestQA(question, passage):
    
    #For accraucy in API request add question mark to question if there isnt one
    if question[-1] != '?':
        question = question + "?"

    #Setup openai API key and retrieve the text-divinci-003
    openai.api_key = ""
    openai.Model.retrieve("text-davinci-003")

    #Request to the api to answer the question based on the given passage
    prom = "Answer the question: " + question + " Based on the text: " + passage
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=prom,
        max_tokens=3000
            )
    
    #return the response of the questions that was asked
    return (resp['choices'][0]['text'])
