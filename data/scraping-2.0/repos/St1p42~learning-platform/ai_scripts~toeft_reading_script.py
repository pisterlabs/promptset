import openai
import re
import json

def getToeflTest():
    openai.api_key = "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"

    passage_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are a skilled AI capable of generating a title and a detailed, academic article between 500-700 words within the realms of humanities, social sciences, life sciences, or physical sciences.The passage should be continuous and it should not specify the names of the subsections"
            },
            {
                "role": "user",
                "content": "Generate a title and corresponding academic passage."
            },
        ],
        temperature=1,
        max_tokens=5000
    )

    title_and_passage = passage_response['choices'][0]['message']['content']


    # Generate the questions and answers
    qa_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are a skilled AI capable of generating TOEFL style comprehension questions and answers from a given passage."
            },
            {
                "role": "user",
                "content": f"Based on the following passage, generate 10 reading comprehension questions each with 4 answer choices, and provide an 'Answer Key'.\n\n{title_and_passage}"
            },
        ],
        temperature=1,
        max_tokens=5000
    )

    question_answer = qa_response['choices'][0]['message']['content']

    json_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are a skilled AI capable of taking a given title, passage, questions, and answer key and generating a structured output similar to a JSON."
            },
            {
                "role": "user",
                "content": f"Take the following title, passage, questions, and answer key and format them into a structured output similar to a JSON. Do not shorten anything. The Json should have these parameters: title, passage, questions: question, options, answer(which contains only the letter). Do not shorten anything f:\n\n{title_and_passage}\n\n{question_answer}"
            },
        ],
        temperature=1,
        max_tokens=12000
    )

    json_like_output = json_response['choices'][0]['message']['content']
    try:
        # Remove unnecessary characters such as new lines or spaces at the beginning and end
        json_like_output = json_like_output.strip()

        # Convert the string to JSON
        json_object = json.loads(json_like_output)
    except json.JSONDecodeError:
        print("Decoding JSON has failed")

    return json.dumps(json_object)





