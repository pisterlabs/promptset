import openai
import asyncio
import subprocess
import re
import os

# Your OpenAI API key
openai.api_key = "API KEY"


def is_safe_command(command):
    # Implement safety checks here
    unsafe_patterns = ["rm ", "sudo", ":(){:|:&};:"]
    return not any(pattern in command for pattern in unsafe_patterns)


def run_shell_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return (
            result.stdout
            or "Command executed successfully, but no output was returned."
        )
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"


def get_full_prompt(user_prompt):
    my_path = os.path.abspath(__file__)
    prompt_path = os.path.dirname(my_path)
    prompt_file = os.path.join(prompt_path, "newprompt.txt")
    pre_prompt = open(prompt_file, "r").read()
    # pre_prompt = pre_prompt.replace("{shell}", shell)
    # pre_prompt = pre_prompt.replace("{os}", get_os_friendly_name())
    prompt = pre_prompt + user_prompt

    # if prompt[-1:] != "?" and prompt[-1:] != ".":
    #     prompt += "?"

    return prompt


def main():
    chat_log = ""
    chat_log += get_full_prompt(chat_log)
    while True:
        # print("Chat Log:\n", chat_log)
        # chat_log += get_full_prompt(chat_log)
        user_input = input("You: ")
        chat_log += "You: " + user_input + "\n"

        prompt = chat_log + "AI (proposing a command):"
        # prompt = chat_log + get_full_prompt(user_input)
        # chat_log += get_full_prompt(user_input)

        # prompt = get_full_prompt(user_input) this works reasonably well however it cannot access previous elements of chat log
        # Get response from GPT-4
        response = openai.Completion.create(
            engine="text-davinci-003", prompt=prompt, max_tokens=100
        )

        command = response.choices[0].text.strip()
        chat_log += "AI: " + command + "\n"

        print("Proposed Command:", command)
        confirm = input("Press Enter to execute, or type anything to skip: ")

        if confirm == "" and is_safe_command(command):
            output = run_shell_command(command)
            print("Output:", output)
            chat_log += "Output: " + output + "\n"
        elif confirm != "":
            print("Command skipped.")
        else:
            print("Unsafe command detected, not executing.")


if __name__ == "__main__":
    main()


# async def stream_response(prompt):
#     async with openai.AsyncCompletion(engine="text-davinci-003") as ac:
#         async for response in ac.create(prompt=prompt, max_tokens=100, stream=True):
#             message = response.get("choices", [{}])[0].get("text", "").strip()
#             if message:
#                 print("AI:", message)
#                 return message  # Or handle it as per your application's logic


# async def main():
#     chat_log = ""
#     chat_log += get_full_prompt(chat_log)
#     while True:
#         user_input = input("You: ")
#         chat_log += "You: " + user_input + "\n"

#         prompt = chat_log + "AI (proposing a command):"

#         # Handle the streaming response
#         command = await stream_response(prompt)
#         chat_log += "AI: " + command + "\n"

#         print("Proposed Command:", command)
#         confirm = input("Press Enter to execute, or type anything to skip: ")

#         if confirm == "" and is_safe_command(command):
#             output = run_shell_command(command)
#             print("Output:", output)
#             chat_log += "Output: " + output + "\n"
#         elif confirm != "":
#             print("Command skipped.")
#         else:
#             print("Unsafe command detected, not executing.")


# if __name__ == "__main__":
#     asyncio.run(main())
