from openai import OpenAI

def prompt(prompt):
    print("Producer Agent")
    print(prompt)
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert assistant that can do anything. Please produce the requested output."},
        {"role": "user", "content": prompt}
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


if __name__ == "__main__":

    prompt("I need a python script that will take a list of numbers and return the sum of all the numbers in the list.")
    prompt("I need a set of lyrics for my rap album about Python Data Structures.")