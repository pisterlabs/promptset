import os
import openai
from datetime import datetime
import pyttsx3
import speech_recognition as sr
from config import apikey

def say(text):
    engine = pyttsx3.init(driverName='espeak')  # Linux voice engine
    engine.setProperty('voice', 'm1')
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def save_conversation(convo_id, chat_dict):
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = f"Openai/convo_{convo_id}_{timestamp}.txt"

    with open(filepath, "w") as f:
        f.write(f"User: {chat_dict[convo_id]['user_query']}\nAI: {chat_dict[convo_id]['ai_response']}")

    print(f"Conversation saved to: {filepath}")

def load_conversations():
    chat_dict = {}
    folder_path = 'Openai'

    try:
        # Iterate through files in the Openai folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                # Extract conversation ID and timestamp from the file name
                parts = file_name.split("_")
                convo_id = int(parts[1])
                timestamp = parts[2].split(".")[0]

                # Read the content of the text file
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    content = file.read()

                # Update chat_dict with the conversation details
                chat_dict[convo_id] = {
                    'user_query': content.split("User: ")[1].split("\n")[0].strip(),
                    'ai_response': content.split("AI: ")[1].strip(),
                    'timestamp': timestamp
                }

        return chat_dict

    except FileNotFoundError:
        # Return an empty dictionary if no saved conversations are found
        return {}

def recall_conversation(convo_id, chat_dict):
    if convo_id in chat_dict:
        print(f"Recalling conversation with ID {convo_id}:")
        print(f"User: {chat_dict[convo_id]['user_query']}")
        print(f"AI Response: {chat_dict[convo_id]['ai_response']}")
        say(f"Recalling conversation with ID {convo_id}. You said: {chat_dict[convo_id]['user_query']}. The AI responded: {chat_dict[convo_id]['ai_response']}")
    else:
        print(f"No conversation found with ID {convo_id}")
        say(f"No conversation found with ID {convo_id}")

def chat(query, chat_dict):
    try:
        print(f"API Key: {apikey}")
        openai.api_key = apikey
        # Load existing conversations at the beginning
        chat_dict = load_conversations()

        # Generate a unique ID for the conversation
        convo_id = len(chat_dict) + 1

        # Update chat_dict with the current conversation
        #print("ChatDict After Adding Conversation:", chat_dict)

        chat_dict[convo_id] = {
            'user_query': query,
            'ai_response': '',
            'timestamp': datetime.now().strftime("%Y%m%d%H%M%S")
        }

        # Use a separate variable for saving to a file
        save_convo_id = convo_id

        # Construct the prompt for OpenAI
        prompt = ""
        for c_id, convo in chat_dict.items():
            prompt += f"User: {convo['user_query']}\nAI: {convo['ai_response']}\n"

        # Add the current user query to the prompt
        prompt += f"User: {query}\n"

        #print("Prompt to OpenAI:")
        print(prompt)

        print(f"ID of the current conversation: {convo_id}")
        # print("ChatDict After Adding Conversation:", chat_dict)


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
        say(response_text)

        # Save the conversation to a file using the separate variable
        save_conversation(save_convo_id, chat_dict)

    except Exception as e:
        print(f"An error occurred while interacting with OpenAI: {e}")

# Example usage
# Initialize chat dictionary
chat_dict = load_conversations()

# Print existing conversation IDs
#print("Existing Conversation IDs:", list(chat_dict.keys()))

# Initialize recognizer
recognizer = sr.Recognizer()

# Example usage

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

                        recall_conversation(convo_id, chat_dict)
                        continue  # Skip the rest of the loop to avoid processing as a new query

                # Update chat_dict using the chat function
                chat(recognized_text, chat_dict)

            else:
                print("Sorry, I did not understand what you said.")

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
