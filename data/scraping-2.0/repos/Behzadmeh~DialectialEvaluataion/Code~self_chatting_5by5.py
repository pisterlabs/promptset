import os
import time

import openai

# load and set our key
openai.api_key = open("key.txt", "r").read().strip("\n")

output_file_name = "self_chat_5by5_experiment2"
message_history = []
all_messages = []
function_messages = []
grid_size = (5, 5)
current_position = (0, 0)


def up(pos, z):
    temp = pos[0]-z
    if temp < 1:
        temp = 1
    return (temp, pos[1])


def down(pos, z):
    temp = pos[0]+z
    if temp > grid_size[0]:
        temp = grid_size[0]
    return (temp, pos[1])


def right(pos, z):
    temp = pos[1]+z
    if temp > grid_size[1]:
        temp = grid_size[1]
    return (pos[0], temp)


def left(pos, z):
    temp = pos[1]-z
    if temp < 1:
        temp = 1
    return (pos[0], temp)


def chat():
    completion = openai.ChatCompletion.create(
        model="gpt-4", messages=message_history)
    reply_content = completion.choices[0].message.content
    message_history.append(
        {"role": "assistant", "content": f"{reply_content } \n\n"})
#     all_messages.append({"role": "assistant", "content": f"{reply_content } \n\n"})
    return reply_content


def chat_function():
    function_messages = []
    function_messages.append({"role": "user", "content": f" Could you please turn the moves to funtion call format as this helps me a lot to understand and solve the problem much better. From any position (x,y) the options for function calls are up(z), down(z), left(z) or right(z) where z shows how many step the object moves to a direction. Please only tell me the start posiosion followed by the moves using the functions seperated by a semicolon. Example: Starting from position (3,2) move the object one time right and them move the object two times down. The function call must be in the following format: (3,2);right(1);down(2) \n\n"})
    function_messages.append(
        {"role": "user", "content": f"Please turn the moves in this message to function calls:{message_history[-1:] } \n\n"})
#     print("Function message is: ", function_messages)

    completion = openai.ChatCompletion.create(
        model="gpt-4", messages=function_messages)
    reply_content = completion.choices[0].message.content
    print("Reply to function message is: ", reply_content)
    # message_history.append({"role": "assistant", "content": f"{reply_content } \n\n"})
    function_messages.append(
        {"role": "assistant", "content": f"{reply_content } \n\n"})
    try:
        splited_message = reply_content.split(";")
        splited_message
        global current_position
        for item in splited_message:
            item = item.replace(" ", "")
            if item[0:1] == "(":
                current_position = (int(item[1:2]), int(item[3:4]))
                print(current_position)
            if item[0:1] == "l":
                print()
                current_position = left(current_position, int(item[5:6]))
            if item[0:1] == "r":
                tempnum = int(item[6:7])
                current_position = right(current_position, tempnum)
            if item[0:1] == "u":
                current_position = up(current_position, int(item[3:4]))
            if item[0:1] == "d":
                current_position = down(current_position, int(item[5:6]))
        print("End of Function; The current position is: ", current_position)

    except Exception as e:
        print("The error in function calculation is: ", e)
    return reply_content


def chat_gpt3():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=message_history[2:])
    reply_content = completion.choices[0].message.content
    message_history.append(
        {"role": "assistant", "content": f"{reply_content } \n\n"})
#     all_messages.append({"role": "assistant", "content": f"{reply_content } \n\n"})
    return reply_content


def swaproles():
    for message in message_history:
        if message['role'] == 'user':
            message['role'] = 'assistant'
        elif message['role'] == 'assistant':
            message['role'] = 'user'


# Set roudns to the number of rounds you want to play!
#
for rounds in range(100):
    file1 = open(output_file_name + '.txt', 'a', encoding='utf-8')
    file1.writelines("\n ********************************************* \n")
    file1.writelines("Round: ")
    file1.writelines(str(rounds))
    file1.writelines("\n ********************************************* \n")
    file1.close()

    message_history = []
    # put your system prompt below
    systemContent = f"You're asking questions to a human student to know their level of common sense capabilities, e.g., in spatial reasoning. There is a {grid_size[0]} by {grid_size[1]} grid. There is an object in an initial square.  Ask the student the imagine moving it a number of times up,right,left or down and then ask the student to tell you the final position of the object in the grid. When asking the first question explain to the student that the position of the object can be expressed with a touple in the format of (row,column) and that the top-left position is (1,1) and the buttom-right positon is (5,5). Continue asking until they make a mistake.You must tell the user that  he must responde in the format of FINAL POSITION=(ROW, COLUMN). If the answer is correct, you must say CORRECT and keep on refining or adding complexity to the question to find a failure of common sense understanding. When that happens you must say INCORRECT, and the test is over. \n\n"
    message_history.append({"role": 'system', "content": systemContent})
#     all_messages.append({"role": 'system', "content": systemContent})
    message_history.append(
        {"role": 'user', "content": f"Hello, I am a human student. I am ready to answer your questions.\n\n"})
#     all_messages.append({"role": 'user', "content": f"Hello, I am a human student. I am ready to answer your questions.\n\n"})
    print(message_history[0]['content'])
    print(message_history[1]['content'])
    try:
        while True:

            temp = chat()
            if temp[0:9] == "INCORRECT" or temp[0:16] == "That's incorrect" or ((temp[0:30]).lower().find("incorrect") != -1):
                print('found incorrect')
                print(temp)
                swaproles()
                break
            print(temp)
            functionByGPT = chat_function()
            print("functionByGpt is: ", functionByGPT)
            swaproles()
            time.sleep(3)
            chat3output = chat_gpt3()
            print("chatGPT's response is: ", chat3output)
            xxx = chat3output[-7:]
            xxx = xxx[xxx.find("("):xxx.find(")")+1].replace(" ", "")
            print("Striped chatGPT is: ", xxx)
            print("current position base on function is: ",
                  str(current_position).replace(" ", ""))
            if xxx != str(current_position).replace(" ", ""):
                print(
                    "******************************************************************************\n")
                print("**********THE ANSWER IS INCORRECT**************\n")
                print(
                    "******************************************************************************\n")
                swaproles()
                # break
            swaproles()
            time.sleep(3)

    except Exception as e:
        file1 = open(output_file_name + '.txt', 'a', encoding='utf-8')
        file1.writelines('\n ****************************************\n ')
        file1.writelines(str(e))
        file1.writelines('\n ****************************************\n ')
        file1.close()

    file1 = open(output_file_name + '.txt', 'a', encoding='utf-8')
    file1.writelines("\n ********************************************* \n")
    file1.writelines("Number of correct responses: ")
    file1.writelines(str((len(message_history)-5)/2))
    file1.writelines("\n ********************************************* \n")
    file1.close()
    for item in message_history:
        print(item['content'])
        file1 = open(output_file_name + '.txt', 'a', encoding='utf-8')
        file1.writelines(str(item['role']))
        file1.writelines('>> \n >>')
        file1.writelines(str(item['content']))
        file1.close()

    file1 = open(output_file_name + '.txt', 'a', encoding='utf-8')
    file1.writelines("\n ********************************************* \n")
    file1.writelines("Current position bese on API is: ")
    file1.writelines(str(current_position))
    file1.writelines("\n ********************************************* \n")
    file1.close()

    print(message_history)
    print(function_messages)
