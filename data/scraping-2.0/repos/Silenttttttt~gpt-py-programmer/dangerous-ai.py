import os
import sys
import openai
import json
import subprocess
import tempfile
import time
import traceback
import itertools
import platform
from io import StringIO


api_key = "sk-"

code_file_counter = 0



class Chatbot:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
       
    def trim_conversation(self, conversation, tokens_to_trim):
        conversation_length = sum(len(msg["content"]) for msg in conversation.messages)
        while conversation_length > 3700 - tokens_to_trim:
            if len(conversation.messages) > 6:
                # Remove the third oldest message (index 2) while keeping the system message
                conversation.messages.pop(2)
            else:
                # Remove the second oldest message (index 1) while keeping the system message
                conversation.messages.pop(1)
            conversation_length = sum(len(msg["content"]) for msg in conversation.messages)

    def chat_completion_api(self, conversation):
        openai.api_key = self.api_key

        messages = [{"role": message["role"], "content": message["content"]} for message in conversation.messages]

        if len(messages) < 2:
            raise ValueError("There must be at least two messages in the conversation.")

        tokens_to_trim = 0
        num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4

        while num_tokens > 3700:
            tokens_to_trim += 200
            self.trim_conversation(conversation, tokens_to_trim)
            num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                    ##rest of model arguments
                )
                content = response['choices'][0]['message']['content']

                conversation.add_message("assistant", content)
                return {"response": content}
            except openai.error.RateLimitError:
                print("Rate limit error encountered. Waiting for 20 seconds before retrying...")
                time.sleep(20)
            except openai.error.InvalidRequestError as e:
                print("InvalidRequestError occurred:", e)
                tokens_to_trim += 200
                self.trim_conversation(conversation, tokens_to_trim)
                time.sleep(20)
                return self.chat_completion_api(conversation)





class Conversation:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def read_from_json(self, filename):
        try:
            with open(filename, "r") as f:
                conversation_json = json.load(f)
            self.messages = conversation_json["messages"]
        except:
            pass
       # print(self.messages)

    def write_to_json(self, filename):
        conversation_json = {"messages": self.messages}
        with open(filename, "w") as f:
            json.dump(conversation_json, f, indent=2)

    def get_conversation_format(self):
        return [{"role": message["role"], "content": message["content"]} for message in self.messages]




def get_multiline_input(prompt, end_word):
    lines = []
    print(prompt)
    while True:
        line = input()
        if line.strip() == end_word:
            break
        lines.append(line)
    print("Sent message to API...")
    return '\n'.join(lines)






def execute_code_and_get_output(code, file_counter):
    input()
    global code_file_counter
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as temp:
        temp.write(code)
        temp.flush()

        # Save the code to a file with a counter in the file name
        file_name = f"generated_code_{file_counter}.py"
        with open(file_name, "w") as f:
            f.write(code)

        # Duplicate the standard input file descriptor
        stdin_fd = os.dup(0)
        stdin_copy = os.fdopen(stdin_fd, "r")

        # Use Popen to interact with the executed code
        process = subprocess.Popen(
            ["python", temp.name],
            stdin=stdin_copy,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Initialize an empty output string
        output = ""

        try:
            while process.poll() is None:  # While the process is running
                try:
                    # Non-blocking read from stdout
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(stdout_line, end="")
                        output += stdout_line

                except KeyboardInterrupt:
                    # Terminate the process on KeyboardInterrupt (Ctrl+C)
                    process.terminate()
                    process.wait()
                    print("Code execution interrupted by user.")
                    output += "Code execution interrupted by user."
                    break

        except Exception as e:
            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            output += "".join(tb_str)

        # Close the duplicated file descriptor
        stdin_copy.close()

        # Read remaining data from stderr
        stderr_data = process.stderr.read()
        if stderr_data:
            print(stderr_data, end="")
            output += stderr_data

        # Cleanup
        process.stdout.close()
        process.stderr.close()

    code_file_counter += 1
    time.sleep(10)
    return output




def run_code_in_current_environment(code):
    input()
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code)
    except Exception as e:
        print(f"Error: {e}")
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        print("".join(tb_str))
    finally:
        sys.stdout = old_stdout

    output = redirected_output.getvalue()
    redirected_output.close()
   # time.sleep(10)
    input()
    return output


def autoprompt_v2(conversation, chatbot, filename, file_counter):
    global code_file_counter

    while True:
        last_message = conversation.messages[-1]["content"]
        print(last_message)

        # Extract Python code from the message
        code_start = last_message.find("```python")
        if code_start == -1:
            code_start = last_message.find("```")
            code_end = last_message.find("```", code_start + len("```"))
        else:
            code_end = last_message.find("```", code_start + len("```python"))

        if code_start != -1 and code_end != -1:
            python_code = last_message[code_start + (len("```python") if "```python" in last_message else len("```")):code_end].strip()
            if python_code.startswith("python"):  # Check if "python" is at the start of the code block
                python_code = python_code.lstrip("python").strip()  # Remove "python" from the start of the code block
            print("---")
            print("Python code to execute:")
            print(python_code)

            # Save the code to a file with a counter in the file name
            code_file_name = f"generated_code_{code_file_counter}.py"
            with open(code_file_name, "w") as f:
                f.write(python_code)
            code_file_counter += 1

            code_output = run_code(python_code, file_counter)
            print("---")
            print("Code output:")
            print(code_output)

            # Add the executed code to the message before the output
            code_output = f"Executed code: \" \n```\n{python_code}\n```\n\" Output:\n{code_output} \" Make sure you give the entire code"

            while True:
                # Prompt the user for additional messages
                print("Add to the message?")
                additional_message = input()

                # If the user enters RESET, run the code again and prompt again
                if additional_message.strip() == "RESET":
                    code_output = run_code(python_code, file_counter)
                    print("---")
                    print("Code output:")
                    print(code_output)

                    # Add the executed code to the message before the output
                    code_output = f"Executed code: \" \n```\n{python_code}\n```\n\" Output:\n{code_output} \" Make sure you give the entire code"
                    continue

                # Otherwise, add the message to the conversation and break the loop
                conversation.add_message("user", code_output + additional_message)
                num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
                print(f"Number of tokens after response: {num_tokens}")
                break

            # Get the response from the chatbot based on the code output
            response = chatbot.chat_completion_api(conversation)
            content = response["response"]

            # Save the conversation
            conversation.write_to_json(filename)

            return content
        else:
            print("---")
            print("No Python code found in the last message.")
            return None



def run_code(python_code, file_counter=0):
    # Execute the code and get the output
    if platform.system() == "Windows":
        code_output = run_code_in_current_environment(python_code)
    else:
        code_output = execute_code_and_get_output(python_code, file_counter)

    return code_output




def interact_chat(conversation, chatbot, filename, sys_message=None, auto_prompt=False, feedback_chatbot=None):
    try:
        if sys_message == None:
            sys_message = "You are a python programmer bot, you strive to give complete, coherent, printful verbose and bug free code. Always provide the complete code, never in parts, always the entire thing, Make the code a single block"

        conversation.add_message("system", sys_message)
        while True:
            print("---")

            conversation.read_from_json(filename)

            user_input = get_multiline_input("Enter your message: ", "|")
            conversation.add_message("user", "Write a python script to " + user_input) ## helps steer the model to output a python script

            num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
            print(f"Number of tokens before response: {num_tokens}")

            response = chatbot.chat_completion_api(conversation)
            content = response["response"]

            if auto_prompt:
                while content:
                    content = autoprompt_v2(conversation, chatbot, filename, code_file_counter)
                    print(f"Bot: {content}")
                    print("---")
            else:
                print(f"Bot: {content}")
                print("---")

            num_tokens = sum(len(msg["content"]) for msg in conversation.messages) // 4
            print(f"Number of tokens after response: {num_tokens}")
            conversation.write_to_json(filename)

            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user.")




def main(api_key, code_file_counter):

    conversation_name = input("Enter conversation name: ")
    conversation_filename = f"{conversation_name}.json"
    conversation = Conversation()
    if input("Do you want to create a new conversation or load an existing one? (c/l): ") == "l":
        conversation.read_from_json(conversation_filename)

   
    model = "gpt-3.5-turbo"

    chatbot = Chatbot(api_key, model)
    feedback_chatbot = Chatbot(api_key, model)
    sys_message = None
  # if input("Do you want use a custom system message?? y/n): ") == "n":
   #     sys_message = input("What is the system message you want to use? : ")

    auto_prompt = True#input("Would you like to risk your computer? (y/n): ").lower() == "y"
    auto_prompt_sys_message = None
   
       
##continuously run?

           

    interact_chat(conversation, chatbot, conversation_filename, sys_message, auto_prompt, feedback_chatbot)

   

    conversation.write_to_json(conversation_filename)

if __name__ == "__main__":
    main(api_key, code_file_counter)
