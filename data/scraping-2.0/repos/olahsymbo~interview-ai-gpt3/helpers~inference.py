import os
import openai

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')


def generate_response(input_text):
    response = openai.Completion.create(
        engine="davinci:ft-personal-2023-01-25-19-20-17",
        prompt="The following is a conversation with DSA an AI assistant. "
               "DSA is an interview bot who is very helpful and knowledgeable in data structure and algorithms.\n\n"
               "Human: Hello, who are you?\n"
               "DSA: I am DSA, an interview digital assistant. How can I help you today?\n"
               "Human: {}\nDSA:".format(input_text),
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " DSA:"]
    )
    return response.choices[0].text.strip()


if __name__ == "__main__":
    input_text = "what is breadth first search algorithm"
    output = generate_response(input_text)
    print(output)