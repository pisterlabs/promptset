from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def prompt(content, numQns, difficulty):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": f"""Generate {numQns} multiple-choice questions with 4 options each based on the following article: \
Format the questions in this way: \
    Questions:
    ~question~ 1. [Question 1]
        a. [Option 1]
        b. [Option 2]
        c. [Option 3]
        d. [Option 4]

    ~question~ 2. [Question 2]
        a. [Option 1]
        b. [Option 2]
        c. [Option 3]
        d. [Option 4]

    ...

    Answer Key:
    1. Correct Answer for Question 1
    2. Correct Answer for Question 2
    ...

    Ensure that the questions cover a range of topics and are of {difficulty} difficulty level."""}
        ]
    )
    return completion.choices[0].message.content

