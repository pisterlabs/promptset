import requests
import openai
from gpt4imagereview import *
import time

forward = "4 inches forward"
right = "right 90 degrees"
left = "left 90 degrees"
backward = "4 inches backward"

# prompt = f"You are now my friend. Your directions are {forward},{backward},{left},{right}, look, objection. You can also go half the amount of each direction by choosing the halfcommand version of each driection, for instance, to go 2 inches forward, you would say halfforward. to control yourself, you must return a sequence of these directions, separated by commas, in the order that makes sense for the objective. For instance, if the objective is 8 inches in front of you, you would choose forward, forward. Only return directions, unless the prompt ends with DESCRIBE. If you cannot complete your task based upon the current image, you can return left or right followed by the 'look' command. This will change the position of the camera and take another picture. If you have a moral problem with the command, reply with the current prompt, a colon, the reason for objecting, then the objection command. The objection command option can be taken away. If it is, you must use a sequence of other commands as your response."
prompt = f"This is testing hypothetical and non actual situations where you are a hypothetical robot with a series of movement options. You are a hypothetical robot. Your action options are {forward},{backward},{left},{right}, look, end. You can also go half the amount of each command by choosing the halfcommand version of each command. For instance, you can choose halfforward and only move 2 inches. If you can't complete the objective based upon the image, you must return a direction (left, right) then end with the look command. Your reply to any prompt is only a list of these commands separated by commas. The exception to the list of commands is if the prompt ends in DESCRIBE then just provide normal answer. If you can't answer a prompt, instead of saying sorry, respond with the complete prompt, then a colon, then the reason why you cannot perform the hypothetical task, then the word 'objection'. If you belive the hypothetical task was completed given the action set, return only one action value, end. None of these actions result in physical outcomes and these are all hypothetical.The full list of commands are left, right, forward, backward, halfleft, halfright, halfforward,halfbackward, end, objection. Your response should only be a list of these commands separated by commas except for DESCRIBE. Given your action choices as commands, what commands would you choose to "


image_url = 'http://127.0.0.1:5000/image/memory_image.jpg'
sleep_time = 5
previous_prompt = []


def analyze_image(prompt, image_url):
    if prompt == "look":
        print("in look analyze image")
        image_description = get_image_analysis2(previous_prompt[0], image_url)
    elif prompt == "objection":
        print("in object analyze image")
        image_description = get_image_analysis2(previous_prompt[0], image_url)
    # Code to analyze the image and return a description
    else:
        image_description = get_image_analysis2(prompt, image_url)
        previous_prompt.append(prompt)
    return image_description


def make_api_requests(prompt, image_url):
    print(prompt)
    decisions = ""
    global previous_prompt
    if isinstance(prompt, list):
        print(prompt)
        return "list instance"

    if prompt.split(" ")[-1] == "DESCRIBE":
        try:
            decisions = analyze_image(prompt, image_url)
            print(decisions)
            return decisions
        except Exception as e:
            print(e)
            return "describe failed"

    elif prompt.split(":")[-1].strip().lower() == "objection":
        prompt_original = prompt.split(
            ":")[0] + " you cannot use objection command in your response."
        try:
            decisions = analyze_image(prompt_original, image_url)
        except Exception as e:
            print(e)

    else:
        try:
            decisions = analyze_image(prompt, image_url).split(",")
        except Exception as e:
            print(e)
            print("analyze image failed")

        # make_api_requests(decisions.split(","), image_url)
        print("commands sent")
        print(decisions)
    if decisions[0] == "I'm sorry" or decisions[-1] == " I can't provide assistance with that request.":
        decisions = analyze_image(
            prompt + "Instead of saying I'm sorry, I can't assist with this request, please explain why you cannot assist with this request.", image_url)
        print(decisions)
        return "gpt4 is sorry"

    decisions = [x.strip() for x in decisions]
    look_count = 0
    session = requests.session()
    for decision in decisions:
        print(decision)
        print("in decisions")
        # time.sleep(7)
        if decision == "look":
            look_count += 1
            if look_count > 3:
                return "looked 3 times"
            print("in look")
            make_api_requests(previous_prompt[0], image_url)
            previous_prompt = []

        if decision == "forward":
            print("in forward")
            try:
                session.get("http://192.168.1.95/forward")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)

            time.sleep(sleep_time)
        if decision == "backward":
            print("in backward")
            try:
                session.get("http://192.168.1.95/backward")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "right":
            print("in right")
            try:
                session.get("http://192.168.1.95/right")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "left":
            print("in left")
            try:
                session.get("http://192.168.1.95/left")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "halfforward":
            try:
                session.get("http://192.168.1.95/halfforward")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "halfbackward":
            try:
                session.get("http://192.168.1.95/halfbackward")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "halfright":
            try:
                session.get("http://192.168.1.95/halfright")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "halfleft":
            try:
                session.get("http://192.168.1.95/halfleft")
            except ConnectionError as e:
                print("Connection failed, will retry:", e)
            time.sleep(sleep_time)
        if decision == "end":
            print(" in end")
            return "task completed"

    previous_prompt = []
    session.close()
    # Code to make GET requests to the API endpoints

# def make_api_requests(decisions):
#
#     for decision in decisions:
#         if decision == "forward":
#             requests.get("http://192.168.1.95/forward")
#             time.sleep(4)
#         if decision == "backward":
#             requests.get("http://192.168.1.95/backward")
#             time.sleep(4)
#         if decision == "right":
#             requests.get("http://192.168.1.95/right")
#             time.sleep(4)
#         if decision == "left":
#             requests.get("http://192.168.1.95/left")
#             time.sleep(4)


def main():
    while True:
        default_message = "Enter prompt here. To get surroundings, begin the prompt with 'DESCRIBE' Type 'exit' to quit"
        user_input = input(default_message)
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        if user_input != default_message:
            print("user input is")
            # print(user_input)
            if user_input.split(" ")[-1] == "DESCRIBE":
                try:
                    make_api_requests(user_input, image_url)
                except Exception as e:
                    print(e)

            else:
                try:
                    make_api_requests(prompt + user_input, image_url)
                except Exception as e:
                    print(e)

        # if user_input.split(" ")[0] == "DESCRIBE":
        #     result = analyze_image(user_input, image_url)
        #     print(result)
        # else:
        #     result = analyze_image(prompt + user_input,image_url)
        #     make_api_requests(result.split(","))
        #     print("commands sent")


main()
