import json
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import re


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def summarize_chunks(text_chunks, client, model):
    context = ""
    output = []
    tokens_called = 0
    for chunk in text_chunks:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": context + "using the context of the previous text, summarize the new text: "
                },
                {
                    "role": "user",
                    "content": "Summarize the following text " + chunk[0]
                }
            ]
        )
        # response is an OpenAI object get the text from it
        response = completion.choices[0].message.content
        context = "Here is a summary of the previous text: " + response + " "
        output.append(response)
        tokens_called += completion.usage.total_tokens
        if tokens_called > 60000:
            sleep(60)
            tokens_called = 0
    return output


def clean_json(string):
    string = re.sub(",[ \t\r\n]+}", "}", string)
    string = re.sub(",[ \t\r\n]+\]", "]", string)
    return string


def create_lessons(lesson_chunks, client, model, grade_level, custom_prompt):
    lessons = []
    tokens_called = 0
    system_prompt = "Using the following summary create a lesson plan that helps students understand the text. The lesson plan should be written in a way that is easy for students to understand. Do not include any explanations, only provide a RFC8259 compliant JSON response with the following structure. "
    system_prompt += '''{
        "Title": "The title of the lesson",
        "Objective": "A brief description of the lesson objective",
        "Materials": "A brief description of the materials needed for the lesson",
        "Assessment": "A brief description of how the student will be assessed"
        "Procedure": {
            "Step One": "Procedure step description",
            "Step Two": "Procedure step description",
            "...": "..."
        }
    }'''
    if grade_level != "":
        system_prompt += " The lesson plan should be appropriate for students in the " + \
            grade_level + " grade."
    if custom_prompt != "":
        system_prompt += " " + custom_prompt
    for chunk in lesson_chunks:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
        )
        # turn the response into a json object
        clean_content = clean_json(completion.choices[0].message.content)
        lesson = json.loads(clean_content)
        lessons.append(lesson)
        tokens_called += completion.usage.total_tokens
        if tokens_called > 60000:
            sleep(60)
            tokens_called = 0
    return lessons


def create_chunks_from_string(string, encoding_name, chunk_size):
    chunks = []
    chunk = ""
    for word in string.split(" "):
        if num_tokens_from_string(chunk + word, encoding_name) > chunk_size:
            chunks.append(
                (chunk, num_tokens_from_string(chunk, encoding_name)))
            chunk = ""
        chunk += word + " "
    chunks.append((chunk, num_tokens_from_string(chunk, encoding_name)))
    return chunks

# Grade level should be a string that is "K, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12"
def gpt_caller(input_object, grade_level="", custom_prompt=""):
    # load API key from .env file
    load_dotenv()
    client = OpenAI()
    model = "gpt-3.5-turbo-1106"
    model_token_limit = 16000

    lessons = []
    for k in input_object:
        grade_level = k
        all_text = input_object[k]
        # add all the texts together
        text = ""
        for t in all_text:
            text += all_text[t] + "\n"
        # remove newlines
        text = text.replace("\n", " ")
        # split text into chunks of a little less than half the token limit
        text_chunks = create_chunks_from_string(
            text, model, model_token_limit / 2.2)

        # Summarize each chunk and use previous summary as context for next chunk
        output = summarize_chunks(text_chunks, client, model)

        # Add up chunks from outputs such that each chunk is less than 2 / 3 of the token limit
        lesson_chunks = []
        chunk = ""
        for summary in output:
            if num_tokens_from_string(chunk + summary, model) > model_token_limit * (2 / 3):
                lesson_chunks.append(chunk)
                chunk = ""
            chunk += summary + " "
        lesson_chunks.append(chunk)

        # Now create a lesson plan based on the summary
        lessons += create_lessons(lesson_chunks, client, model, grade_level, custom_prompt)

    lessons = [json.dumps(lesson) for lesson in lessons]

    return lessons