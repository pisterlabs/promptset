version=1.1
import os
import sys
import time
import requests
import keyboard
import subprocess
from tiktoken_ext import openai_public
import tiktoken_ext
import tiktoken
from plyer import notification
import pystray
from PIL import Image

import win32con
from win32gui import GetWindowText, GetForegroundWindow
import win32gui, win32com.client
import win32clipboard


supported_extensions_path = "C:/Users/asdaFemboy/PycharmProjects/githubAutoCommit/supported_extensions.txt"
shell=win32com.client.Dispatch("WScript.Shell")
EXE = r'C:\Users\asdaFemboy\AppData\Local\GitHubDesktop\GitHubDesktop.exe'
git_executable = r'C:\Program Files\Git\cmd\git.exe'


def notify(text):
    notification.notify(
        # title of the notification,
        title="AI Commit Writer",
        # the body of the notification
        message=text, )


def select_window_by_name(window_name):
    exe_path = EXE

    subprocess.run(exe_path)

    # window_handle = win32gui.FindWindow(None, window_name)
    # if window_handle == 0:
    #     print(f"Window '{window_name}' not found.")
    # else:
    #     print(window_handle)
    #     window_name = win32gui.GetWindowText(window_handle)
    #     print("Window Name:", window_name)
    #     shell.SendKeys(' ')  # Undocks my focus from Python IDLE
    #     win32gui.SetForegroundWindow(window_handle)
    #     win32gui.BringWindowToTop(window_handle)


def is_git_installed():
    try:
        result = subprocess.run([git_executable, '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def on_keypress(event):
    if event.event_type == keyboard.KEY_DOWN:
        if keyboard.is_pressed('ctrl') and keyboard.is_pressed('alt') and keyboard.is_pressed('c'):
            select_window_by_name("GitHub Desktop")
            time.sleep(0.5)

            if GetWindowText(GetForegroundWindow()) == "GitHub Desktop":
                do_commit()
            else:
                print("failed to select github desktop")


def type_and_submit(var1, var2):
    time.sleep(0.5)
    keyboard.press_and_release('ctrl+g')

    time.sleep(0.5)  # Wait for the target application to process the Tab keypresses

    # title
    keyboard.write(var1)
    keyboard.press_and_release('tab')
    # summary
    keyboard.write(var2)

    keyboard.press_and_release('tab')
    keyboard.press_and_release('tab')
    keyboard.press_and_release('enter')


def send_request(request):
    url = "https://free.churchless.tech/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": request
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    content = response.json()['choices'][0]['message']['content'].strip('"')
    # print(content)
    return content


def get_filepath():
    # laptop
    # keyboard.press_and_release("ctrl+'")
    # pc
    keyboard.press_and_release("ctrl+`")
    time.sleep(1)
    active_window_handle = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(active_window_handle)

    if window_title == "Command Prompt":
        # win 11 uses ctrl shift a
        # keyboard.press_and_release('ctrl+shift+a')
        keyboard.press_and_release('ctrl+a')
        time.sleep(0.1)
        keyboard.press_and_release('ctrl+c')
        time.sleep(0.1)

        win32clipboard.OpenClipboard()
        clipboard_text = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT)
        win32clipboard.CloseClipboard()
        clipboard_text = clipboard_text.decode('utf-8')  # Assuming UTF-8 encoding

        win32gui.PostMessage(active_window_handle, win32con.WM_CLOSE, 0, 0)

        cleaned_text = clipboard_text.split(") ", 1)[-1]
        # Remove the trailing ">"
        cleaned_text = cleaned_text.strip()
        cleaned_text = cleaned_text[:-1]

        return cleaned_text


def filter_text_files(file_list):
    # Read the supported extensions from a text file
    with open(supported_extensions_path, 'r') as f:
        supported_extensions = f.read().splitlines()

    # Filter the file list to include only text files
    filtered_files = [file.strip() for file in file_list if file.strip().lower().endswith(tuple(supported_extensions))]

    return filtered_files


def get_changed_files(repo_path):
    os.chdir(repo_path)
    result = subprocess.run(['cd', repo_path, '&&', git_executable, 'status', '--porcelain'], shell=True,
                            capture_output=True)

    # Check if the command encountered an error
    if result.returncode != 0:
        # An error occurred
        print("An error occurred while executing the command.")
        print("Error Message:", result.stderr.decode().strip())
        return []

    # Parse the output to extract the file names
    changed_files = []
    for line in result.stdout.splitlines():
        # line = line.encode('utf-8')
        if line.startswith((b' M ', b'?? ', b' D ', b'A  ')):
            file_name = line[2:].decode('utf-8')
            file_name = file_name.replace('"', "")
            if not file_name.startswith(' '):
                file_name = ' ' + file_name
            changed_files.append(file_name)

    return changed_files


def get_individual_changes(file_path, changed_files):
    individual_changes = {}

    # Iterate over the changed files
    for file in changed_files:
        # Search for the file within the file_path and its subfolders
        for root, dirs, files in os.walk(file_path):
            if file in files:
                # Construct the absolute path to the file
                file_abs_path = os.path.join(root, file)

                # Enclose the file path in double quotation marks
                file_abs_path_quoted = f"{file_abs_path}"

                # Change working directory to file_path
                os.chdir(file_path)

                # Execute the git diff command to retrieve the changes
                cmd = ['git', 'diff', file_abs_path_quoted]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=file_path)

                # Check if the command encountered an error
                if result.returncode != 0:
                    # An error occurred
                    print("An error occurred while executing the command.")
                    print("Error Message:", result.stderr)
                    return []

                # Extract the + and - lines from the git diff output
                changes = [line for line in result.stdout.splitlines() if
                           line.startswith(('+', '-')) and not line.startswith(('--- a/', '+++ b/'))]

                # Store the changes in the dictionary with the file name as the key
                if (changes) == []:
                    changes = [
                        '+   This file  ' + file + " Is completely new. Please make sure to Say that you created it"]
                individual_changes[file] = changes

                break  # Stop searching once the file is found

    return individual_changes


def convert_list_to_lines(input_list):
    lines = '\n'.join(input_list)
    return lines


def count_tokens(input_text, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_text))

    return num_tokens


def gpt_summary(changes, file_path, token_limit):
    summaries = []

    for file, changes_list in changes.items():
        # Search for the file within the file_path and its subfolders
        for root, dirs, files in os.walk(file_path):
            if file in files:
                # Construct the absolute path to the file
                file_abs_path = os.path.join(root, file)

                # Read the original file text
                with open(file_abs_path, 'r') as f:
                    original_text = f.read()

                # Format the original file and changes into the summary string
                summary = f"Here is the original file for context \n {original_text} \n and here are the changes \n {', '.join(changes_list)} \n, summarize this in the shortest way possible, the next object is below which you must also summarize. If the changes do not say it was created do not say it was created"

                # Check token count
                token_count = count_tokens(summary, "cl100k_base")
                if token_count > token_limit:
                    summaries.append(
                        "Large changes made to file: " + file + " Just say that large changes were made. Do not make up changes which I have not told you")
                    print("Summary exceeds token limit.")
                else:
                    summaries.append(summary)

                break  # Stop searching once the file is found

    return summaries


def split_summaries(summaries, token_limit):
    split_lists = []
    current_list = []
    current_token_count = 0

    for summary in summaries:
        token_count = count_tokens(summary, "cl100k_base")
        if current_token_count + token_count > token_limit:
            split_lists.append(current_list)
            current_list = []
            current_token_count = 0

        current_list.append(summary)
        current_token_count += token_count

    if current_list:
        split_lists.append(current_list)

    return split_lists


def gpt_requester(split_lists):
    final_response = None

    prompt = 'Below are several files, each file contains the original content (usually text, numbers and symbols) along with the changes made to it. Please summarise the changes in each file as shortly as you can, making sure to say the file name. Please only say the summaries and nothing else. Write your responses in this style "Created a new file named current scores.txt" or "Added a new method: Start() that logs "has started"".  Do not write in a list. Do not skip any files. If a change starts with + the line was added. If it starts with - it was removed. If a section is removed say so. Thank you:\n'

    for split_list in split_lists:
        # Combine the summaries in the split list
        combined_summary = prompt + '\n'.join(split_list)

        # Check token count of combined summary
        token_count = count_tokens(combined_summary, "cl100k_base")
        if token_count > 4096:
            # Split the combined summary into new split lists
            new_split_lists = split_summaries(split_list, 4096)
            # Recursively call gpt_requester for the new split lists
            response = gpt_requester(prompt + new_split_lists)
        else:
            # Send the request for the combined summary
            notify("Beginning generation")
            response = send_request(combined_summary)
            print(combined_summary)
            notify("Generation completed")

        # Add the response to the final response list
        if final_response is None:
            final_response = response
        else:
            final_response += response

    # Summarize the final response as shortly as possible
    final_summary = "Summarize all the responses as shortly as possible:\n" + final_response
    token_count = count_tokens(final_summary, "cl100k_base")
    while token_count > 4096:
        # Split the final summary into new split lists
        split_lists = split_summaries(final_response, 4096)
        final_response = gpt_requester(split_lists)
        final_summary = "Summarize all the responses as shortly as possible:\n" + final_response
        token_count = count_tokens(final_summary, "cl100k_base")

    return final_response


def do_commit():
    print("commiting")
    if is_git_installed():
        file_path = get_filepath()
        changed_files = filter_text_files(get_changed_files(file_path))

        print(get_changed_files(file_path))

        print("repo path: " + file_path)
        print("changed files: ")
        print(changed_files)

        changes_dict = get_individual_changes(file_path, changed_files)

        # # Print the changes for each file
        # for file, changes in changes_dict.items():
        #     print("File:", file)
        #     print("Changes:")
        #     for change in changes:
        #         print(change)
        #     print()

        initial_summary = gpt_summary(changes_dict, file_path, 4096)
        splitted_summaries = split_summaries(initial_summary, 4096)
        final_summary = (gpt_requester(splitted_summaries))

        print("final summary:")
        print(final_summary)

        type_and_submit("Various changes", final_summary)
    else:
        print("Git not installed or not on filepath")


keyboard.on_press(on_keypress)
keyboard.wait()

# get all the changed files (done)
# remove any not ending in plaintext file extensions like png (done)
# use git diff and then the path to them to see the changes (done)

# check to see if the file length is over 2048 chars, if it is we will mark this file as unsuitable and its summary
# will be "changed" if it's less than 2048 we will give the file to chat gpt saying here is the original file,
# here are my changes, summarise these as shortly as possible. Specify the original text is just there for context,
# in some cases there will be no context, and it will just need to summarise the changes this will then be the summary
# for each file, associate it with the file using a key pair dictionary. The key is the file name the pair is the
# summary finally index all of that into 1 big prompt asking for the shortest summary physically possible. Use this
# as the description

# instead of using char limit use tokens, https://platform.openai.com/tokenizer, limit is 4096 tokens. Submit the
# code with all indentation removed since that is irrelevant to function
