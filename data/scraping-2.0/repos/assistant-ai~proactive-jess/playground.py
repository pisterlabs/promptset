from openai import OpenAI

import time
import json

client = OpenAI()

JESS_NAME = "Jess"
JESS_ID = "asst_5rHDVLiJ8N1JBfRUIoFUreLo"

def read_jeff_instructions():
    with open("jess_prompt.txt") as f:
        return f.read()
    return ""

instructions = read_jeff_instructions()

function_to_schedule_message = {
    "type": "function",
    "function": {
        "name": "schedule_message",
        "description": "Schedule next message to be sent to a user if there is not response from user after a defined period of time.",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "integer", 
                    "description": "Amount of second from now when the message should be sent."
                },
                "message": {
                    "type": "string", 
                    "description": "Message to be sent to the user."
                },
            },
            "required": ["message", "time"]
        }
    }
} 

if instructions == "":
    print("Please create a file called instructions.txt and put your instructions in it.")
    exit(1)

jess = None

for assistant in client.beta.assistants.list():
    if assistant.id == JESS_ID:
        jess = assistant
        print("Found assistant")
        break

jess_assitent_args = {
    "name": JESS_NAME,
    "instructions": instructions,
    "tools": [function_to_schedule_message],
    "model": "gpt-4-1106-preview"
}

if not jess:
    print("Creating new assistant")
    jess = client.beta.assistants.create(
        **jess_assitent_args
    )
else:
    print("Updating assistant")
    jess = client.beta.assistants.update(
        assistant_id=jess.id,
        **jess_assitent_args
    )

print("Assistant created/updated")
default_thread = client.beta.threads.create(
  messages=[]
)

def send_system_message(message_to_send):
    send_message(message_to_send, role="user")

def send_message(message_to_send, role="user"):
    message = client.beta.threads.messages.create(
        thread_id=default_thread.id,
        role=role,
        content=message_to_send
    )

    run = client.beta.threads.runs.create(
        thread_id=default_thread.id,
        assistant_id=jess.id
    )

    print("checking assistant status. ")
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=default_thread.id,
            run_id=run.id
        )

        print(run.status)

        if run.status == "completed":
            print("done!")
            messages = client.beta.threads.messages.list(
                thread_id=default_thread.id
            )

            print("messages: ")
            for message in messages:
                print({
                    "role": message.role,
                    "message": message.content[0].text.value
                })        
            break
        if run.status == "requires_action":
            messages = client.beta.threads.messages.list(
                thread_id=default_thread.id
            )

            print("messages: ")
            for message in messages:
                print({
                    "role": message.role,
                    "message": message.content[0].text.value
                }) 
            print("requires_action!")
            print(str(run.required_action))
            call_id = run.required_action.submit_tool_outputs.tool_calls[0].id
            arg_string = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
            # parse string with json into python dict
            args = json.loads(arg_string)
            arugment_time = args["time"]
            arugment_message = args["message"]
            print("time: " + str(arugment_time))
            print("message: " + arugment_message)
            # client.beta.threads.runs.cancel(
            #     run_id=run.id,
            #     thread_id=default_thread.id
            # )
            time.sleep(5)
            print("sending message: " + arugment_message)
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=default_thread.id,
                run_id=run.id,
                tool_outputs=[
                    {
                        "tool_call_id": call_id,
                        "output": "done"
                    }
                ]
            )
        else:
            print("in progress...")
            time.sleep(5)

send_message("Hi, how are you?")
send_system_message("This is not real user message, and user will not read it. This message allows you to make a decision if you want to schedule a pro-active message to a user. Do it if you think it is required. Message that you might schedule will NOT be sent if the user will answer to you first so do not be afraid of spamming user.")
#send_message("can you ping me in 5 seconds?")
client.beta.threads.delete(default_thread.id)
