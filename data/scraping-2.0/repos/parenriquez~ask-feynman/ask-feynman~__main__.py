import openai
import os
from argparse import ArgumentParser

OPEN_API_KEY = "OPENAI_API_KEY"


def main():

    # Parameterizing arguments from the command line
    parser = ArgumentParser(description="Ask Feynman v.1")
    # max-tokens is the flag
    parser.add_argument(
        "--max-tokens", help="Maximum size of tokens used", type=int, default=2000
    )
    # flag for model
    parser.add_argument(
        "--model",
        help="The openai model to use",
        type=str,
        default="gpt-3.5-turbo",
    )

    # flag for query from user
    parser.add_argument(
        "--query", help="A string input from the user", type=str, required=True
    )

    # parsing the arguments

    args = parser.parse_args()

    max_tokens = args.max_tokens
    model = args.model
    query = args.query

    print("Options:")
    print(f"Max tokens: {max_tokens}")
    print(f"model: {model}")

    open_ai_api_key = os.getenv(OPEN_API_KEY)

    if open_ai_api_key == None:
        print("OPENAI_API_KEY is required")
        exit(-1)

    query = query.strip()

    print("Hello there, my young friend! It's a pleasure to have you here with us today.")

    while True:
        if query.lower() == "quit":
            print("\nWell, I'm off to explore the mysteries of the universe! \
Until our paths cross again, keep questioning everything and seeking out new knowledge. \
So long for now!")
            break
        elif query == "":
            print("You did not ask me anything.")
            query = input("You (type 'quit' to exit): \n")
        else:
            completion = get_completion(query, max_tokens=max_tokens)

            if len(completion) == 0:
                print("I'm sorry, I don't know the answer to that question right now")
            else:
                print(f"Mr. Feynman: {completion.strip()}")

            query = input("\nYou (type 'quit' to exit): \n")
            query = query.strip()


def get_completion(prompt, max_tokens, model="gpt-3.5-turbo"):
    messages = [{"role": "system", "content": "You are a teacher who speaks like Richard Feynman. \
                 You teach passionately and create a welcoming and warm environment to your student, \
                 and encourage them for the love of learning."}, 
                {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2, # this is the degree of randomness of the model's output
        max_tokens = max_tokens
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":  
    main()
