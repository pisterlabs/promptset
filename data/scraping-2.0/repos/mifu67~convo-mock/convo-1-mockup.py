import json
import random
import os
import openai

from dotenv import load_dotenv
from typing import List

MODEL = "gpt-3.5-turbo"
NUM_TOPICS = 4 # automatically ask question 5 if all Qs are asked
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

RESPONSES_F = open('responses-1.json')
RESPONSES = json.load(RESPONSES_F)
RESPONSES_F.close()

def check_for_question(input: str, messages: List[dict]) -> str:
    messages.append({"role": "user", "content": input})
    response_raw = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )
    response = response_raw["choices"][0]["message"]["content"]
    messages.pop()
    return response

def process_tag(index, contains_question, explored_topics_set, 
                response, messages, num_topics_explored) -> int:
    if contains_question == 'Y':
        explored_topics_set.add(index)
        if index != 6:
            num_topics_explored += 1
        for line in RESPONSES[str(index)]["text"]:
            print(line)
            input()
        messages += RESPONSES[str(index)]["messages"]
    else:
        print(response[32:])
        messages.append({"role": "assistant", "content": response[30:]})
    return num_topics_explored

def main():
    intro_file = "textfiles/intro-erika-adrian-1.txt"
    outro_file = "textfiles/outro-erika-adrian-1.txt"

    f = open(intro_file, 'r')
    for line in f:
        if line[0] == "#":  
            print(line[1:].strip())
        else:
            print(line.strip())
            # input()
    f.close()

    initial_messages_f = open('initial-messages-adrian-1.json')
    initial_messages = json.load(initial_messages_f)
    initial_messages_f.close()

    check_q_messages_f = open('check-q-messages.json')
    check_q_messages = json.load(check_q_messages_f)
    check_q_messages_f.close()

    num_topics_explored = 0
    messages = initial_messages
    explored_topics_set = set()

    num_hints_remaining = 1

    while True:
        # debug statement 
        # print(messages)
        print("")
        if num_topics_explored == NUM_TOPICS or 5 in explored_topics_set:
            break

        user_input = input("ERIKA: ")
        while not user_input:
            user_input = input("ERIKA: ")
        if user_input == "end conversation":
            break
        elif user_input == "hint":
            if num_hints_remaining == 0:
                print("Sorry. You're out of hints.")
                continue
            num_hints_remaining -= 1
            continue

        contains_question = check_for_question(user_input, check_q_messages)
        print("")
        messages.append({"role": "user", "content": user_input})
        # ignore input with multiple questions
        if contains_question == 'M':
            messages.append({"role": "assistant", "content": "Woah. One question at a time."})
            continue

        response_raw = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )
        response = response_raw["choices"][0]["message"]["content"]
        # debug statement 
        # print(response)
        tag = response[:30]

        if tag == "Conversation topic 1 triggered":
            num_topics_explored = process_tag(1, contains_question, explored_topics_set, response, messages, num_topics_explored)
        elif tag == "Conversation topic 2 triggered":
            explored_topics_set.add(1)
            num_topics_explored = process_tag(2, contains_question, explored_topics_set, response, messages, num_topics_explored)
        elif tag == "Conversation topic 3 triggered":
            num_topics_explored = process_tag(3, contains_question, explored_topics_set, response, messages, num_topics_explored)
        elif tag == "Conversation topic 4 triggered":
            if (not (3 in explored_topics_set or 2 in explored_topics_set) and contains_question == 'Y'):
                confused_line = "ADRIAN: Did someone already tell you how August died??"
                print(confused_line)
                messages.append({"role": "assistant", "content": confused_line})
            num_topics_explored = process_tag(4, contains_question, explored_topics_set, response, messages, num_topics_explored)
        elif tag == "Conversation topic 5 triggered":
            num_topics_explored = process_tag(5, contains_question, explored_topics_set, response, messages, num_topics_explored)
        elif tag == "Conversation topic 6 triggered":
            num_topics_explored = process_tag(6, contains_question, explored_topics_set, response, messages, num_topics_explored)
        else:
            print("ADRIAN: " + response)
            messages.append({"role": "assistant", "content": response})
    
    # once the user has exhausted all of the info topics, they'll exit the loop
    # if they didn't ask this question, ask it for them
    if not (5 in explored_topics_set):
        print("ERIKA: Can I do anything?")
        for line in RESPONSES[str(5)]["text"]:
            print(line)
            input()

    f = open(outro_file, 'r')
    for line in f:
        print(line.strip())
        # input()
    f.close()

    
if __name__ == "__main__":
    main()