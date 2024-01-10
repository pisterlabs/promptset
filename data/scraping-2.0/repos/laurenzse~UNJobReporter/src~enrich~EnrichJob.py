import json
import os

import openai
import tiktoken
from dotenv import load_dotenv

from src.data.Job import UNJob

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def answer_job_questions(job: UNJob, question_function: dict):
    prompt = f"""
        A job was posted with the following description:
        
        -----------------------------------------------

        {job.description}
        
        -----------------------------------------------

        Please summarize and filter out this job using a JSON data structure. 
        
        Your responses should be brief as space is limited and the user needs to be able to quickly scan the answers.
                
        If the description uses terminology, use the same terminology in your answer.
        
        Along with your answers, the user will see key facts about the job posts, e.g. the job title, the organization, the location etc.
        Do not repeat such facts in your answers but rather extract and summarize facts out of the job description, which
        the user otherwise might miss.
        """

    prompt_tokens = num_tokens_from_string(prompt, "gpt-3.5-turbo")
    function_tokens = num_tokens_from_string(
        json.dumps(question_function), "gpt-3.5-turbo"
    )
    answer_tokens = function_tokens * 1.5

    if prompt_tokens + function_tokens + answer_tokens > 16384:
        print("Prompt too long.")
        return None
    elif prompt_tokens + function_tokens + answer_tokens > 4096:
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-3.5-turbo"

    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful job search assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            functions=[question_function],
        )
    except Exception as e:
        print(f"An error occurred while prompting GPT: {e}")
        return None

    try:
        generated_text = completion.choices[0].message.function_call.arguments
        return json.loads(generated_text)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
