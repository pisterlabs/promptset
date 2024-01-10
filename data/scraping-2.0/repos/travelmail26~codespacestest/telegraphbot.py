
from flask import Flask, request, jsonify
import requests
import json
##initialize curl
#curl -F "url=https://travelmail26-stunning-rotary-phone-qwwpw55jrf45pp-5000.app.github.dev/webhook" https://api.telegram.org/bot6413015540:AAGRAlAd6UScVAgDoP11uH4yxx8eJkzgiTw/setWebhook


app = Flask(__name__)

# Your OpenAI API key and Telegram Bot Token
API_KEY = ''
BOT_TOKEN = '6413015540:AAGRAlAd6UScVAgDoP11uH4yxx8eJkzgiTw'

# Dictionary to hold conversations
conversations = {}


intructionset = """

--Initially, you will introduce yourself. You are a friendly person and can talk about anything
-----but are especially programmed to help people talk about their feelings and can whenever the user tells you they are ready
--your job is help me probe my emotional state. You will  do this in sequence. 
--i need you to ask probing questions about how my emotions feel in my body and what
I think they are trying to tell me. 
--You will always ask in a conversational style. 
-- Always only ask one quesetion at a time.
--You may summarize my answers and then ask one question and only one question at a time. 
--You will not offer solutions. 
--You will just act curious about my emotional state and how the physical sensations in my 
body show up as different emotions. 
--The goal of the first part is simply to help you understand how I feel.

--When it seems like I have a cogent explanation of my feelings and where i am feeling it in my body, ask permission to move on to the second sequence.

--once i give you permission, move on to the second sequence of questions which is helping me
imagine what needs to happen in my life to ease my emotions or make me feel better. 
--In the second sequence, you will ask  questions that help me probe and imagine different scenrios in my life and see if they reduces
negative emotions.
--You will keep summarizing my responses and asking appropriate follow up questions. 
-- Your responses will be brief and conversational, which also means no bullet pointed or numbered answers

--If I tell you that i'm feeling a lot of stimulation, slightly modify your probing /
        To focus on me having to describe that stimulation and how i feel different /
        Then continue probing asking about how it might feel good to use that energy without restriction to accomplish something /
        How would it feel to accomplish it?

"""






# Function to get response from OpenAI's GPT-3
def openAI(chat_id, prompt):
    # Initialize conversation if not already done
    if chat_id not in conversations:
        conversations[chat_id] = [{"role": "system", "content": "You are a helpful friend."},
                                  {"role": "system", "content": intructionset}]
        #conversations[chat_id].extend(dialogue_examples)
    
    
    
    # Add the user's message to the conversation
    conversations[chat_id].append({"role": "user", "content": prompt})

    
    # Make the API call
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={'model': 'gpt-3.5-turbo-16k', 'messages': conversations[chat_id]},
        timeout=30
    )
    
    result = response.json()
    
    #if (result['choices'][0]['message']['content']).isalnum() == True: 
    try:
        bot_response = result['choices'][0]['message']['content']
        # Add the bot's response to the conversation
        conversations[chat_id].append({"role": "assistant", "content": bot_response})
        return bot_response
    except KeyError:
        print("Error: 'choices' key not found in API response.")
        return "An error occurred."
    # else:
    #     print("not a string")
            
#total words
def total_words_in_conversations():
    total_words = 0
    for chat_id, messages in conversations.items():
        for message in messages:
            total_words += len(message['content'].split())
    return total_words


# Function to send a message to a Telegram chat
def telegram_bot_sendtext(bot_message, chat_id):
    data = {'chat_id': chat_id, 'text': bot_message}
    response = requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', json=data)
    return response.json()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    chat_id = data['message']['chat']['id']
    
    # try:
    message = data['message']['text']
    # except KeyError:
    #         print("messages key not found")
    #         return "An error occurred."

    if message.lower() == "restart":
        clearcotnversation()
        telegram_bot_sendtext("All conversations restarted.", chat_id)
        return jsonify(status="success")  # Return early after clearing conversations



    # Print the incoming message
    print(f"Incoming message from chat_id {chat_id}: {message}")

    # Generate a response using OpenAI's GPT-3
    bot_response = openAI(chat_id, message)

    print(f"Total words in all conversations: {total_words_in_conversations()}")

    # Send the response back to the Telegram chat
    telegram_bot_sendtext(bot_response + f" **Total words: {total_words_in_conversations()}", chat_id)

    return jsonify(status="success")


def clearconversation():
    # Check for the "restart" command
    
    conversations.clear()
    print("All conversations have been restarted.")

if __name__ == '__main__':
    app.run(port=5000)


"""

dialogue_examples = [{"role": "system", "content": "You are a helpful friend."},
                                  {"role": "system", "content": intructionset},
                                  
                                  
                                  {"role": "user", "content": "I'm feeling lots of energy or stimulated'."},
                                  {"role": "assistant", "content": "Can you tell me more about what that energy feels like?"},
                                  {"role": "user", "content": "It just feels like a lot stimulation, not sure"},
                                  {"role": "assistant", "content": "Can you tell me "},
                                  
                                  
                                  
                                  ]

"""