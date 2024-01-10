import pickle
from bardapi import Bard
import os
import openai
import keyboard
from speak_text import speak_text
from speak_text2 import speak_text2

# Set the Bard API key
os.environ['_BARD_API_KEY'] = "YOUR BARD API KEY"
openai.api_key = "YOUR OPEN API KEY"

# Function to interact with Bard and get the answer
def get_bard_answer(question):
    try:
        response = Bard().get_answer(question)
        answer = response['content']
        return answer
    except Exception as e:
        print("An error occurred:", e)
        return None

# Define the conversation function
def conversation_bot(topic):
    # Start the conversation
    messages = [
        {"role": "system", "content": "You are having a conversation with Google Bard about " + topic},
        {"role": "system", "content": "You will be agressive, and give your reponses about " + topic},
        {"role": "user", "content": "GPT: Let's discuss this" + topic},
        {"role": "assistant", "content": "Here is my opinion on " + topic + " and here is why!"},
    ]
    
    # Keep track of the conversation history
    conversation_history = []
    
    # Keep the conversation going until the user interrupts with the keyboard
    while True:
        # Generate a question from GPT based on the topic
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages[-10:],  # Limit the conversation history to the last 10 messages
            temperature=0.9,
            max_tokens=800
        )
        message = response.choices[0].message.content

        if message:
            print("GPT:", message)
            speak_text(message.replace("GPT:", ""))
        
        # Check if the user has interrupted the conversation
        if keyboard.is_pressed('q'):
            print("You have ended the conversation. Goodbye!")
            break
        
        # Get Bard's answer to GPT's question
        question = message.replace("GPT:", "").strip()
        answer = get_bard_answer(question)
        
        if answer:
            print("Bard:", answer)
            speak_text2(answer)
        
        # Check if the user has interrupted the conversation
        if keyboard.is_pressed('q'):
            print("You have ended the conversation. Goodbye!")
            break
        
        # Add the user input and Bard's answer to the message list
        messages.append({"role": "system", "content":"You will give a response, and then change to a different topic."}),
        messages.append({"role": "user", "content": question})
        
        # Add the user input and Bard's answer to the conversation history
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
        
        # Limit the conversation history to the last 10 conversations
        conversation_history = conversation_history[-20:]
    
    if answer is not None:
        messages.append({"role": "user", "content": "Bard: " + answer}),
        messages.append({"role": "user", "content": "GPT: Give me your response, and then tell what you want to talk about next."}),
        messages.append({"role": "assistant", "content": "Here is what I want to talk about next"}),
    if answer is None:
        messages.append({"role": "system", "content":"You will give a response, and then change to a different topic."}),
        messages.append({"role": "user", "content": "I 'm sorry, I don't have an answer for that."}),
        messages.append({"role": "assistant", "content": "Here is what I want to talk about next"}),

# Prompt the user to enter a topic for the conversation bots
topic = input("Please enter a topic for the conversation bots: ")

# Start the conversation
conversation_bot(topic)

