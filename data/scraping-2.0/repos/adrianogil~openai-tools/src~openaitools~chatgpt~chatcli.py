import datetime
import openai
import json
import os


openai.api_key = os.environ["CHATGPT_API_KEY"]

data_environ_var = "CHATGPT_CHAT_BKP_DIR"
default_chat_data_folder = os.environ[data_environ_var] if data_environ_var in os.environ else os.path.join(os.environ["HOME"], ".chatgpt")

today = datetime.datetime.now()
year_folder = today.strftime('%Y')
month_folder = today.strftime('%m')
week_folder = today.strftime('%Y.%mW%V')
time_file_name = today.strftime('chat-%Y.%m.%d-%Hh%M')
file_name = f"{time_file_name}.json"

folder_path = os.path.join(default_chat_data_folder, year_folder, month_folder, week_folder)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

note_data = {
    "subject": "",
    "messages": [],
    "prompt_index": 0,
    "markdown_data": f"""

# CHAT - {today.strftime("%Y.%m.%d-%Hh%M")}

"""
}



def get_chatgpt_output(user_input="Hello world!"):
    note_data["messages"].append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=note_data["messages"])
    chatgpt_output = completion.choices[0].message.content
    note_data["messages"].append({"role": "assistant", "content": chatgpt_output})
    
    if note_data["subject"]:
        file_name = f"{time_file_name} - {note_data['subject']}.json"
    else:
        file_name = f"{time_file_name}.json"
    data_path = os.path.join(folder_path, file_name)
    with open(data_path, 'w') as file_handler:
         json.dump({"messages": note_data["messages"]}, file_handler)


    note_data["prompt_index"] += 1
    note_data["markdown_data"] += f"""
## Prompt {"%03d" % (note_data["prompt_index"],)}

{user_input}

### Answer

{chatgpt_output}

    """

    return chatgpt_output


def save_markdown_note():
    if "CHAT_GPT_MARKDOWN_NOTE_DIR" not in os.environ:
        print("ERROR: Failed to save Markdown Note. Please set a CHAT_GPT_MARKDOWN_NOTE_DIR environment var")
        return

    note_folder_path = os.path.join(os.environ["CHAT_GPT_MARKDOWN_NOTE_DIR"], year_folder, month_folder, week_folder)
    note_time_file_name = today.strftime('Chat - %Y.%m.%d-%Hh%M')
    note_file_name = f"{time_file_name}.md"

    if not os.path.exists(note_folder_path):
        os.makedirs(note_folder_path)

    with open(os.path.join(note_folder_path, note_file_name), 'w') as file_handler:
        file_handler.write(note_data["markdown_data"])
    print("Note saved!")


def start_chat_loop():
    user_input = input("> ").strip()
    while user_input not in ["exit", "q"]:
        if user_input.startswith("set subject as "):
            note_data["subject"] = user_input[15:]
        elif user_input == "save as note":
            save_markdown_note()
        elif user_input.startswith("save file as "):
            save_file_name = user_input[13:]
            with open(save_file_name, 'w') as file_handler:
                file_handler.write(chatgpt_output)
            print(f"Last message saved as {save_file_name}")
        else:
            chatgpt_output = get_chatgpt_output(user_input)
            print(chatgpt_output)
        user_input = input("> ").strip()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        chat_file = sys.argv[1]
        file_name = os.path.basename(chat_file)
        with open(chat_file, 'r') as file_handler:
            chat_data = json.load(file_handler)
            note_data["messages"] = chat_data["messages"]
            for message in note_data["messages"]:
                if message["role"] == "user":
                    print(">", message["content"])
                else:
                    print(message["content"])
        if len(sys.argv) > 2:
            attachment_file = sys.argv[2]
            with open(attachment_file, 'r') as file_handler:
                lines = file_handler.readlines()
            note_data["messages"].append({
                "role": "user",
                "content": f"I'll present a file in path '{attachment_file}' with the content below. \n\n" + \
                    "\n".join(lines)
            })
    start_chat_loop()    
