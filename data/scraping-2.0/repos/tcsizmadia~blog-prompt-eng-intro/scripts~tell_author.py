# getting Ollama support from langchain
from langchain.llms import Ollama

# initializing Ollama with Llama2 moddel with 7b parameters
ollama = Ollama(model="llama2:7b")


# a simple function to ask the model to tell the author of a book,
# given the title of the book
def tell_author(book_title):
    prompt = f"Tell me the author of the book: '{book_title}'!"
    reply = ollama(prompt, temperature=0.1, seed=2)
    return reply


# asking the user for a book title
title = input("Give me a book title: ")

# storing the answer in a variable, for brevity
answer = tell_author(title)

# telling the user the answer
print(answer)
