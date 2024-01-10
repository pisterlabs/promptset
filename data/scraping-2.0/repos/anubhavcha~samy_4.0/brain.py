import wikipedia
import webbrowser
import time
import sys
import threading
from openaibrain import chat_with_ai



# Make sure you have the required modules and functions for speaking
from speak import speak

def load_qa_data(file_path):
    qa_dict = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                continue
            q, a = parts
            qa_dict[q] = a
    return qa_dict


def save_qa_data(file_path, qa_dict):
    with open(file_path, "w", encoding="utf-8") as f:
        for q, a in qa_dict.items():
            f.write(f"{q}:{a}\n")


qa_file_path = "C:\\Users\\Anubhav\\OneDrive\\Desktop\\Samy_3.0\\qna_logbook.txt"
qa_dict = load_qa_data(qa_file_path)

def openai_brain(text):
    try:
        prompt = text
        response = chat_with_ai(prompt)

        if not response:
            wiki_search(text)  # If OpenAI response is empty, pass to wiki_search
            return

        # Start animation and speaking concurrently
        animate_thread = threading.Thread(target=print_animated_message, args=(response,))
        speak_thread = threading.Thread(target=speak, args=(response,))

        animate_thread.start()
        speak_thread.start()

        animate_thread.join()
        speak_thread.join()

        # Assuming 'search_text' is defined somewhere
        qa_dict[text] = response  # Store in qa_dict
        save_qa_data(qa_file_path, qa_dict)  # Save updated qa_dict
    except Exception as e:
        # Handle any other exception
        print(f"An error occurred: {str(e)}")
        wiki_search(text)  # Pass to wiki_search on error

def wiki_search(text):
    search_text = text.replace("jarvis", "")
    search_text = search_text.replace("wikipedia", "")

    try:
        wiki_summary = wikipedia.summary(search_text, sentences=2)
        animate_thread = threading.Thread(target=print_animated_message, args=(wiki_summary,))
        speak_thread = threading.Thread(target=speak, args=(wiki_summary,))

        animate_thread.start()
        speak_thread.start()

        animate_thread.join()
        speak_thread.join()

        # Assuming 'search_text' is defined somewhere
        qa_dict[search_text] = wiki_summary  # Store in qa_dict
        save_qa_data(qa_file_path, qa_dict)  # Save updated qa_dict
    except wikipedia.exceptions.DisambiguationError as e:
        speak("There is a disambiguation page for the given query. Please provide more specific information.")
        print("There is a disambiguation page for the given query. Please provide more specific information.")
    except wikipedia.exceptions.PageError:
        google_search(text)

def google_search(query):
    query = query.replace("who is ", "")
    query = query.strip()

    if query:
        url = "https://www.google.com/search?q=" + query
        webbrowser.open_new_tab(url)
        speak("You can see search results for " + query + " in Google on your screen.")
        # Commenting out the speak function as it's not provided here
        print("You can see search results for " + query + " in Google on your screen.")
    else:
        speak("I didn't catch what you said.")
        # Commenting out the speak function as it's not provided here
        print("I didn't catch what you said.")

def print_animated_message(message):
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.075)  # Adjust the sleep duration for the animation speed
    print()
def search_brain(text):
    if text in qa_dict:
        ans = qa_dict[text]
        animate_thread = threading.Thread(target=print_animated_message, args=(ans,))
        # Create a thread for speaking the message
        speak_thread = threading.Thread(target=speak, args=(ans,))

        # Start both threads concurrently
        animate_thread.start()
        speak_thread.start()

        # Wait for both threads to finish
        animate_thread.join()
        speak_thread.join()
    else:
        openai_brain(text)
