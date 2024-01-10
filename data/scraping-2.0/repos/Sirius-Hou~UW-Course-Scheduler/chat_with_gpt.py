import os
import sys
import openai
import json
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("API_KEY")

def chat_with_gpt3_once():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input("What do you want to ask GPT-3? ")},
    ]
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    print(response['choices'][0]['message']['content'])


"""
chat_with_gpt3(): This function is the main function that will be used to chat with GPT-3.5 turbo.
"""
def chat_with_gpt3():
    print("Welcome to ChatGPT 3.5 turbo! Type 'exit' to quit.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    while True:
        user_input = input(">>> ")
        messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            stop=None,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        print(response['choices'][0]['message']['content'] + "\n")

        if user_input.lower() == "exit":
            break   


#chat_with_gpt3()



def gpt_generate_schedule():
    with open("docs/command.txt", "r") as f:
        command_text = f.read()

    with open("docs/all_courses_info.txt", "r") as f:
        all_courses_info = f.read()

    with open("docs/sample_schedule.txt", "r") as f:
        sample_schedule = f.read()

    # Replace the string in the messages list
    messages = [
        {"role": "user", "content": command_text + "\n" + all_courses_info}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        stop=None,
        messages=messages
    )
    #messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print(response['choices'][0]['message']['content'] + "\n")\
    
    with open("docs/generated_schedule.txt", "w") as f:
        f.write(response['choices'][0]['message']['content'])


#gpt_generate_schedule()


def gpt_convert_to_calendar_format():
    with open("docs/convert_to_calendar_format_cmd.txt", "r") as f:
        convert_to_calendar_format_cmd = f.read()

    with open("docs/generated_schedule.txt", "r") as f:
        generated_schedule = f.read()

    # Replace the string in the messages list
    messages = [
        {"role": "user", "content": convert_to_calendar_format_cmd + "\n" + generated_schedule}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        stop=None,
        messages=messages
    )
    #messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print(response['choices'][0]['message']['content'] + "\n")
    
    with open("docs/generated_schedule_calendar_format.txt", "w") as f:
        f.write(response['choices'][0]['message']['content'])
    

gpt_convert_to_calendar_format()