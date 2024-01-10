import os
import openai

# Load OpenAI API key from environment
openai.api_key = os.env("OPENAI_API_KEY")

# Set the maximum token limit for GPT-4
TOKEN_LIMIT = 4000


def startup():
    choice = input(
        "What would you like to review:\nEnter 1 for file.\nEnter 2 for chunk of code.\n"
    )
    response = ""
    if choice == "1":
        response = "file"
    else:
        response = "chunk of code"
    print(f"Great! I am going to review your {response}!")

    if choice == "1":
        filename = input("Provide file path: ")
        with open(filename, "r") as f:
            contents = f.read()
        return repr(contents)

    elif choice == "2":
        print("Enter/Paste your content. Ctrl-D to save it.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            contents.append(line)
        return contents

    else:
        print("Wrong Choice, terminating the program.")


def send_chunk_to_openai(chunk):
    print("Calling OpenAI...")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a software engineering assistant, skilled in NextJS, JavaScript, TypeORM, Postgres, TypeScript and GraphQL. You specialize in suggesting code improvements.",
            },
            {
                "role": "user",
                "content": f"Your responsibility is to review the provided code and suggest code improvements in a concise way {chunk}",
            },
        ],
        temperature=0,
        max_tokens=1024,
    )
    print("Done with call to OpenAI.")

    return completion.choices[0].message["content"]


def review_suggestion(suggestion):
    print("Opening VScode for you to review the suggestions")
    with open("openai-code-review.tsx", "w") as f:
        f.write(suggestion)
    os.system("code openai-code-review.tsx")


def main():
    """
    The main function orchestrates the operations of:
    1. Accepting file changes as input
    2. Sending those files to OpenAI for review
    3. Allowing you to review suggestions in VScode
    """

    contents = startup()
    review = send_chunk_to_openai(contents)
    review_suggestion(review)


if __name__ == "__main__":
    main()  # Execute the main function
