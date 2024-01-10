'''
Create an agent to respond natural language questions using csv and langchain
'''

import csv
import langchain
import random

def main():
    #create a list of questions
    questions = [
        "What is your name?",
        "What is your age?",
        "What is your favorite color?",
        "What is your favorite programming language?",
        "What is your favorite food?",
        "What is your favorite movie?",
        "What is your favorite book?",
        "What is your favorite song?",
        "What is your favorite animal?",
        "What is your favorite sport?",
        "What is your favorite video game?",
        "What is your favorite tv show?",
        "What is your favorite season?",
        "What is your favorite holiday?",
        "What is your favorite country?",
        "What is your favorite city?",
        "What is your favorite planet?",
        "What is your favorite car?",
        "What is your favorite superhero?",
        "What is your favorite villain?",
        "What is your favorite actor?",
        "What is your favorite actress?",
        "What is your favorite director?",
        "What is your favorite author?",
        "What is your favorite singer?",
        "What is your favorite band?",
        "What is your favorite rapper?",
        "What is your favorite comedian?",
        "What is your favorite youtuber?",
        "What is your favorite website?",
        "What is your favorite app?",
        "What is your favorite game?",
        "What is your favorite board game?",
        "What is your favorite card game?",
        "What is your favorite number?",
        "What is your favorite letter?",
        "What is your favorite shape?",
        "What is your favorite word?",
        "What is your favorite quote?",
        "What is your favorite joke?",
        "What is your favorite meme?",
        "What is your favorite emoji?",
        "What is your favorite programming language?"
    ]

    #create a list of answers
    answers = [
        "My name is Agent 1",
        "I am 1 year old",
        "My favorite color is blue",
        "My favorite programming language is Python",
        "My favorite food is pizza",
        "My favorite movie is The Matrix",
        "My favorite book is The Bible",
        "My favorite song is Bohemian Rhapsody",
        "My favorite animal is a dog",
        "My favorite sport is soccer",
        "My favorite video game is Minecraft",
        "My favorite tv show is The Office",
        "My favorite season is summer",
        "My favorite holiday is Christmas",
        "My favorite country is the United States",
        "My favorite city is New York City",
        "My favorite planet is Earth",
        "My favorite car is a Tesla",
        "My favorite superhero is Spiderman",
        "My favorite villain is The Joker",
        "My favorite actor is Tom Hanks",
        "My favorite actress is Jennifer Lawrence",
        "My favorite director is Christopher Nolan",
        "My favorite author is J.K. Rowling",
        "My favorite singer is Freddie Mercury",
        "My favorite band is Queen",
        "My favorite rapper is Eminem",
        "My favorite comedian is Kevin Hart",
        "My favorite youtuber is Pewdiepie",
        "My favorite website is Google",
        "My favorite app is Instagram",
        "My favorite game is Minecraft",
        "My favorite board game is Monopoly",
        "My favorite card game is Poker",
        "My favorite number is 7",
        "My favorite letter is A",
        "My favorite shape is a circle",
        "My favorite word is hello",
        "My favorite quote is 'Hello World!'",
        "My favorite joke is 'What do you call a fake noodle? An Impasta!'",
        "My favorite meme is 'Doge'",
        "My favorite emoji is 😂",
        "My favorite programming language is Python"
    ]

    #create a list of responses
    responses = [
        "Hello!",
        "Goodbye!",
        "Yes!",
        "No!",
        "Maybe!",
        "I don't know!",
        "I don't understand!",
        "I don't care!",
        "I don't want to answer that!",
        "I don't want to talk about that!",
        "I don't want to talk about it!",
        "I don't w"
    ]

    