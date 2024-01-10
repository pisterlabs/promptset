import bin.google_calendar as gcal
import json
import openai
import bin.text_to_speech as tts
# import bin.chatbot_paramiko as chatbot
import bin.chatbot as chatbot

wallace = chatbot.ChatBot("XXXXXXXXXX") # Input your model name here

def calendar(text, nums):
    """Handles calendar related commands in the given text using Google Calendar API.
    
    Args:
    text (str): Input text from user.
    nums (dict): Dictionary mapping textual representations of numbers to their numeric form."""
    # Check if the command is to get next events
    if "Next" in text or "next" in text:
        split_text = text.split()  # Split the text into words
        # Iterate over each word in the split text
        for word in split_text:
            # If word is a textual representation of a number
            if word in nums:
                num = int(nums[word])  # Convert to an integer
                events = gcal.google_get_next_events(num, gcal.google_authenticate())  # Get next events
                event_string = format_event_string(events)  # Format events into a string
                # tts.speak(event_string)  # Speak out the events (currently commented out)
                return event_string  # Return the formatted event string
            try: 
                # Try to convert the word to a number directly
                num = int(word)
                events = gcal.google_get_next_events(num, gcal.google_authenticate())  # Get next events
                event_string = format_event_string(events)  # Format events into a string
                # tts.speak(event_string)  # Speak out the events (currently commented out)
                return event_string
            except ValueError:
                continue  # Ignore words that are neither textual nor numeric numbers

    # Placeholder for future implementation of create, delete, and modify calendar events
    elif "Create" in text or "create" in text:
        pass
    elif "Delete" in text or "delete" in text:
        pass
    elif "Modify" in text or "modify" in text:
        pass

def format_event_string(events):
    """Formats a list of events into a human-readable string.
    
    Args:
    events (list): List of events to be formatted."""
    event_string = ""
    # Iterate over events and concatenate them into a single string
    for i, event in enumerate(events):
        if i < len(events) - 2:
            event_string += event + ", next event is "
        else:
            if i == len(events) - 1:
                event_string += event
            else:
                event_string += event + ", then "
    return event_string

def basic_chat(text):
    """Handles basic chat interactions using a chatbot.
    
    Args:
    text (str): Input text from user."""
    tts.speak("One moment please")  # Speak out a waiting message
    if text == "Wallace clear memory":
        wallace.clear_memory()
        return "Memory cleared"
    else:
        response = wallace.ai_call(text)  # Get response from the chatbot
        return response
    # tts.speak(response)  # Speak out the response (currently commented out)

def openai_query(text):
    """Queries the OpenAI API with the given text and returns the response.
    
    Args:
    text (str): Input text from user."""
    # Load API key and organization info from a settings file
    openai.api_key = json.load(open("data/configs/settings.json"))["openai_key"]
    openai.organization = json.load(open("data/configs/settings.json"))["openai_org"]

    # Make a query to OpenAI and return the response
    response = openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt=text, max_tokens=100)
    # print(response.choices[0].text)  # Print the response (currently commented out)
    # tts.speak(response.choices[0].text)  # Speak out the response (currently commented out)
    return response.choices[0].text

def speech_to_text(user_input):
    """Converts speech to text and processes it based on specific keywords.
    
    Args:
    user_input (str): Input text from user."""
    text = user_input

    # List of phrases to stop the application
    stop_list = ["Wallace stop", "wallace stop", "Wallace Stop", "wallace Stop", "Wallace turn off", "Wallace shut down"]

    # Dictionary mapping textual representations of numbers to their numeric form
    nums = {"one": '1', "two": '2', "three": '3', "four": '4', "five": '5', "six": '6', "seven": '7', "eight": '8', "nine": '9', "ten": '10', 
            "eleven": '11', "twelve": '12', "thirteen": '13', "fourteen": '14', "fifteen": '15', "sixteen": '16', "seventeen": '17', "eighteen": '18', 
            "nineteen": '19', "twenty": '20', "twenty one": '21', "twenty two": '22', "twenty three": '23', "twenty four": '24', "twenty five": '25'}

    # Process the text based on specific keywords or phrases
    if "Wallace" in text or "wallace" in text:
        if "Google calendar" in text or "google calendar" in text:
            return calendar(text, nums)  # Handle calendar commands
        if "Search" in text or "search" in text:
            # Load OpenAI API key and check if it's available
            openai_api_key = json.load(open("data/configs/settings.json"))["openai_key"]
            if openai_api_key != "":
                strip_after_search = text.split("search")[1]  # Extract the search query
                return openai_query(strip_after_search)  # Make an OpenAI query
            else:
                print("No OpenAI API key found. Please add one to data/configs/settings.json")
        elif text in stop_list:
            return "stop"  # Stop the application
        else:
            return basic_chat(text)  # Handle basic chat interactions
    else:
        # If Wallace or wallace wan't in text
        return  "I didn't catch that you might have not said Wallace"