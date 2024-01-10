import speech_recognition as sr
import pyttsx3
import openai
import os
from datetime import datetime
from config import apikey

# OpenAI code
def chat(query, chat_dict):
    try:
        openai.api_key = apikey

        print("All Conversation IDs:")
        for convo_id in chat_dict:
            print(convo_id)
        # Generate a unique ID for the conversation
        convo_id = len(chat_dict) + 1

        #print("ChatDict Before Adding Conversation:", chat_dict)

        # Update chat_dict with the current conversation
        chat_dict[convo_id] = {
            'user_query': query,
            'ai_response': '',
            'timestamp': datetime.now().strftime("%Y%m%d%H%M%S")
        }

        #print("ChatDict After Adding Conversation:", chat_dict)

        # Construct the prompt for OpenAI
        prompt = ""
        for cid, convo in chat_dict.items():
            prompt += f"User: {convo['user_query']}\n{convo['ai_response']}\n"

        # Add the current user query to the prompt
        prompt += f"User: {query}\n"

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct-0914",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Update the chat_dict with the AI's response
        chat_dict[convo_id]['ai_response'] = response['choices'][0]['text']

        # Print and say the response
        response_text = chat_dict[convo_id]['ai_response']
        print(f"AI: {response_text}")

        say(chat_dict[convo_id]['ai_response'])

        # Save the conversation to a file
        save_conversation(convo_id, chat_dict)

    except Exception as e:
        print(f"An error occurred while interacting with OpenAI: {e}")

def save_conversation(convo_id, chat_dict):
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = f"Openai/convo_{convo_id}_{timestamp}.txt"

    with open(filepath, "w") as f:
        f.write(f"User: {chat_dict[convo_id]['user_query']}\nAI: {chat_dict[convo_id]['ai_response']}")

    print(f"Conversation saved to: {filepath}")

# Recall conversation function
def recall_conversation(convo_id, chat_dict):
    if convo_id in chat_dict:
        print(f"Recalling conversation with ID {convo_id}:")
        print(f"User: {chat_dict[convo_id]['user_query']}")
        print(f"AI Response: {chat_dict[convo_id]['ai_response']}")
        say(f"Recalling conversation with ID {convo_id}. You said: {chat_dict[convo_id]['user_query']}. The AI responded: {chat_dict[convo_id]['ai_response']}")
    else:
        print(f"No conversation found with ID {convo_id}")
        say(f"No conversation found with ID {convo_id}")

# def get_user_input():
#     while True:
#         user_input = input("Enter conversation ID or 'stop' to exit: ")
#         if user_input.lower() == 'stop':
#             return None
#         try:
#             convo_id = int(user_input)
#             return convo_id
#         except ValueError:
#             print("Invalid input. Please enter a valid conversation ID.")

# ... (remaining code remains unchanged)
def say(text):
    engine = pyttsx3.init(driverName='espeak')  # Linux voice engine
    engine.setProperty('voice', 'm1')
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Initialize recognizer
recognizer = sr.Recognizer()

# Initialize chat dictionary
chatDict = {}

#print("Existing Conversation IDs:", list(chatDict.keys()))
# Example usage
while True:
    with sr.Microphone() as source:
        print("Please start speaking...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Listen to the speech and store it in audio_text variable
            audio_text = recognizer.listen(source)
            print("Audio recording complete.")

            # Use Google Speech Recognition to convert audio to text
            recognized_text = recognizer.recognize_google(audio_text)

            if recognized_text:
                print("User: " + recognized_text)

                # Check if the user wants to stop the conversation
                if recognized_text.lower() == "stop":
                    print("Conversation stopped.")
                    break  # Exit the loop

                # Check if the user wants to recall a past conversation
                elif recognized_text.lower().startswith("recall id"):
                    parts = recognized_text.lower().split("recall id ")
                    if len(parts) == 2:
                        convo_id = int(parts[1].strip())

                        recall_conversation(convo_id, chatDict)
                        continue  # Skip the rest of the loop to avoid processing as a new query

                # Update chatDict using the chat function
                chat(recognized_text, chatDict)

            else:
                print("Sorry, I did not understand what you said.")

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        # Check if the user wants to manually input the conversation ID
        # manual_input = get_user_input()
        # if manual_input is not None:
        #     recall_conversation(manual_input, chatDict)
