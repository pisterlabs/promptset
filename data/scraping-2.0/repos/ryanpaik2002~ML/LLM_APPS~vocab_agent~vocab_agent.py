import openai

import os
from dotenv import load_dotenv
load_dotenv()
# OPENAI_API_KEY = os.getenv("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")


# Set up your OpenAI API credentials
openai.api_key = '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'

def generate_question(word):
    """
    prompts a question for a given word
    missing - prompt setup
    missing -> multple choice response
    TODO: add prompt setup from previous setup
    
    
    """
    prompt = f"Word: {word}\nQuestion"
    

    # Generate multiple-choice question completion using OpenAI's GPT-3 model
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the generated question and answer from the API response
    choices = response.choices[0].text.strip().split('\n')
    print(f"choices: {choices}")
    question = choices[0]
    print(f"question: {question}")
    answer = choices[1]
    print(f"answer: {answer}")
    
    return question, answer

# Example usage
word = "lustrous"
question, answer = generate_question(word)
print(f"Question: {question}")
print(f"Answer: {answer}")

print(answer)