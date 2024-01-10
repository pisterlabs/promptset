import pyperclip
import time
import os
import openai

def clipboard_listener():
    last_clipboard_value = ""
    
    while True:
        clipboard_value = pyperclip.paste()
        if clipboard_value != last_clipboard_value:
            last_clipboard_value = clipboard_value
            chat_gepity(clipboard_value)
        time.sleep(1)


def chat_gepity(clipboard_value):
    print(f"Copied to clipboard: {clipboard_value}")
    openai.organization = "org-CUdodZ48VjZcXBFRtTrlRuFw"
    openai.api_key = "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"
    # print(openai.Model.list())
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in system design and analysis and the following multiple choice question by picking one of the choices."},
                {"role": "user", "content": clipboard_value}
            ]
        )
        
        answer = response['choices'][0]['message']['content']
        print(f"Answer: {answer}")
        print("\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    clipboard_listener()
