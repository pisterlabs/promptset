import os
import openai
import pprint
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv
load_dotenv()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def send_and_receive(messages):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

def message_loop(file):
    words = []
    lineCount = 0

    #define system message
    system_message = {'role':'system'}
    system_message['content'] = "System: You are a filter and error corrector. You simply take the text from the user and provide the list of corrected words and phrases according to the specifications."

    #define user message
    user_message = {'role':'user'}
    user_message['content'] = """
    User: For the following list, 
    extract all english words and phrases, 
    fix any spelling and grammar mistakes. 
    Keep any text that represents optional variants, 
    such as things in brackets, 
    and anything that represents a possible continuation, such as a tilde. 
    Additionally, remove anything that is not english, or is only for formating, 
    including BSHWG, NC, NH, OW, SS. 
    Seperate each english word to a new line if it is not a phrase. 
    Keep phrases on one line. 
    If no english words or phrases are found, 
    return 'No english words or phrases found.
    Do not add any extra details or notes, Only the list.'\n\n"""

    # get next 50 lines from file
    for i in range(50):
        line = file.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        # add line to user message
        user_message['content'] += line
    
    #define message list
    messages = [system_message, user_message]
    completion = send_and_receive(messages)
    # isolate the response
    response = completion['choices'][0]['message']['content']
    # print the response
    print(response)

    if response != "No english words or phrases found.":
        words.append(response)

    # add the response to the message list
    messages.append({'role': 'assistant', 'content': response})

    default_messages = messages.copy()

    input("Press Enter to continue...")
    endfile = False
    
    while not endfile:
        messages = default_messages.copy()

        #define user message
        user_message = {'role':'user'}
        user_message['content'] = "User: "

        # get next 50 lines from file
        for i in range(50):
            line = file.readline()
            # if line is empty, end of file is reached
            if not line:
                endfile = True
                break
            # add line to user message
            lineCount += 1
            user_message['content'] += line

        #define message list
        messages.append(user_message)

        # send message and receive response
        completion = send_and_receive(messages)
        # isolate the response
        response = completion['choices'][0]['message']['content']
        # print the response
        print(response)
        print("Line Count: " + str(lineCount))
        
        # if response is not "No english words or phrases found." add response to words list
        if response == "No english words or phrases found.":
            continue
        words.append(response)
        
        
    
    # write words list to file
    with open("output.txt", "w", encoding="utf-8") as file:
        for word in words:
            file.write(word + "\n")
    
    

def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]


    # get file for input
    file = open("jp_text_words.txt", "r", encoding="utf-8")

    #start message loop
    message_loop(file)

    


if __name__ == '__main__':
    main()