import json
import os
from datetime import datetime
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 

from dotenv import load_dotenv

load_dotenv()  # Before importing openai to set OPENAI_API_BASE



FIRST_TEMP=0.3

# PROBLEMATIC LLMS
# dolphin2.2-mistral: buggy newer version of dolphin?
# mistral-openorca: trails into garbage
# openchat: buggy?
# alfred needs more than 8GB VRAM
# Yi 7B generates garbage
# deepseek-coder generates garbage

# List of local LLMs
local_llms = [ "mistral", "openhermes2.5-mistral", "neural-chat", "zephyr",  "wizard-vicuna-uncensored", "llama2-uncensored", "orca2"]


# Get the current datetime once for the session
session_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

last_llm = None

def llm_log(prompt, llm_name):
    # chat(messages)
    
    if llm_name != last_llm:
        # Initialize Ollama
        llm = Ollama(
            # repeat_penalty=1.7,  # prevent infinite repetitions?
            repeat_last_n=-1, # look at entire context to prevent repetition
            model=llm_name,
            temperature = FIRST_TEMP,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    print(f">>>>>> PROMPT {llm_name}: ", prompt)
    response = llm(prompt) # ollama

    print(f"RESPONSE: {llm_name} ", response)
    session_filename = f"session-{session_datetime}.txt"
    with open(session_filename, "a", encoding="utf-8") as session_file:
        session_file.write(f"\n{'='*20} PROMPT {llm_name} {'='*20}\n")
        session_file.write(prompt)
        session_file.write(f"\n{'='*20} RESPONSE {llm_name} {'='*20}\n")
        session_file.write(response)
    return response


def strip_text_around_json(text):
    # Find the first occurrence of '{' character
    start_index = text.find('{')

    # Find the last occurrence of the '}' character
    end_index = text.rfind('}') + 1  # +1 to include the '}' character

    # Slice the string from start_index to end_index
    stripped_text = text[start_index:end_index]

    return stripped_text


def extract_and_parse_json(string_data):
    stripped_text = strip_text_around_json(string_data)
    parsed_data = json.loads(stripped_text)
    
    # Confirm that there are at least 10 chapters with non-empty 'title' and 'prompt' fields
    steps = parsed_data.get('steps', [])
    if len(steps) < 1:
        raise Exception("Insufficient number of steps.")
        # Ensure 'title' and 'function_name' fields in each chapter are strings
    for step in steps:
        if not isinstance(step.get('function_name'), str):
            raise ValueError(f"'function_name' in step {step.get('step')} is not a string.")
        
        if not isinstance(step.get('prompt'), str):
            raise ValueError(f"'prompt' in step {step.get('step')} is not a string or is missing.")
 

    return parsed_data


def gpt4_llm(prompt):
    return "Not implemented yet"

    # chat(messages)
    response = llm(prompt) # ollama
    parsed_data = extract_and_parse_json(response)
    return parsed_data

# Load questions from JSON file
with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)


# Function to grade an answer using gpt4_llm
def grade_answer(question, answer):
    prompt = f"Grade this answer for the question '{question}' on a scale of 0 to 10, where 10 is perfect: {answer}"
    grade = gpt4_llm(prompt)
    return grade

# Dictionary to store answers and grades for each LLM
llm_answers = {llm_name: [] for llm_name in local_llms}

# Iterate through each LLM and question
for llm_name in local_llms:
    # Set the current model
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", llm_name)

    for question_info in questions:
        question = question_info["question"]
        # Get the answer from the current LLM
        # prompt = f"Answer this succinctly and concisely: {question}\nAnswer:"
        prompt = f"{question}"
        answer = llm_log(prompt, llm_name)

        # Grade the answer
        grade = grade_answer(question, answer)

        # Store the result
        llm_answers[llm_name].append({
            "question": question,
            "answer": answer,
            "grade": grade
        })

# Save the results
results_filename = f"llm-test-results-{session_datetime}.json"
with open(results_filename, "w", encoding="utf-8") as results_file:
    json.dump(llm_answers, results_file, indent=4)

print(f"Test results saved to {results_filename}")
