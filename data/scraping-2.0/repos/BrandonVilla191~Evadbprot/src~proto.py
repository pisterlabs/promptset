import openai
import os 
import evadb 


openai.api_key = "api-key"


def read_codebase(directory):
    return


def generate_solution(prompt):
    return



def main():
    prompt = input("Enter a prompt: ")

    codebase = read_codebase("codebase")

    prompt = prompt + codebase

    solution = generate_solution(prompt)




