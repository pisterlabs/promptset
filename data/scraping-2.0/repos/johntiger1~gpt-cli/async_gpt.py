import openai
import os
import time
import queue
import threading
import keyboard

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up the GPT-3 model
model_engine = "gpt-3.5-turbo"
SKIP_PROMPT = "I have nothing to say right now."

history = [
        {"role": "system", "content": "You are assistantGPT. You are a helpful assistant, who also has the ability to "
                                      "query the user every 10s for any input that you believe "
                                      "may further assist them. If the previous message included the phrase "
                                      "'I have nothing to say right now', then generate the empty response ( ). "
                                      
                                      "}"},
    ]
# Create a message queue to handle user input
message_queue = queue.Queue()

# Define a function to generate a response
def generate_response(prompt):
    global history
    if prompt == SKIP_PROMPT:
        prompt = "I am still thinking of what to ask. Please ask me a clarifying question."
    new_message = {"role": "user", "content": "Please me a great, asynchronous assistant with my daily life. "
                                              "Some common things to ask me are: - have i responded to all my work emails yet?"
                                              "- have I followed up with Jim from accounting on getting those invoices done?"
                                              "- is it anyone's birthday yet? If there is nothing relevant right now, it is always"
                                              "ok to ask me how my day is going and where my head space is."
                                              "\n However, please make it relevant to the conversation we are having."
                                              + prompt
                    }
    history.append(new_message)
    completion = openai.ChatCompletion.create(
        model=model_engine,
        messages=history
    )
    # Command to execute the command
    system_output_response = completion.choices[0].message.content

    output_dict = {"role":"assistant", "content":f"{system_output_response}"}
    history.append(output_dict)
    history.extend(history[:2])
    return system_output_response

def execute_response(response):
    '''in the future, we can do zero-shot toxicity detection '''
    print(response)

def handle_user_input():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            exit(0)
        message_queue.put(user_input)

def handle_message_queue():
    last_user_input_time = 0
    while True:
        if message_queue.qsize() > 0:
            user_input = message_queue.get()
            prompt = user_input
            last_user_input_time = time.time()

        elif time.time() - last_user_input_time >= 5 and not any(keyboard.is_pressed(key) for key in keyboard.all_modifiers):
            prompt = SKIP_PROMPT
            last_user_input_time = time.time()
            print("USER_PROMPT SKIPPED")

        else:
            time.sleep(1)
            continue

        response = str(generate_response(prompt))
        execute_response(response)


if __name__ == "__main__":

    print('Welcome to asyncGPT. Type "exit" to quit at any time')
    user_input = input("You: ")
    prompt = "\nUser: " + user_input
    response = str(generate_response(prompt))
    execute_response(response)

    # Create and start threads to handle user input and message queue
    user_input_thread = threading.Thread(target=handle_user_input)
    message_queue_thread = threading.Thread(target=handle_message_queue)
    user_input_thread.start()
    message_queue_thread.start()

    # Wait for threads to finish
    user_input_thread.join()
    message_queue_thread.join()
