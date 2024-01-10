# Importing Libraries
import telethon # Library to interact with Telegram's API as a user or through a bot account 
from telethon.tl.custom import Button
from telethon import TelegramClient, events

import asyncio # Provides infrastructure for writing asynchronous code using coroutines.
import config # Custom file containing configuration settings for the bot.
import openai # Python module that provides an interface to the OpenAI API.


# Configure OpenAI API key
openai.api_key = config.openai_key

# Configure Telegram client
client = TelegramClient(config.session_name_bot, config.API_ID, config.API_HASH).start(bot_token=config.BOT_TOKEN)


# Define button templates
keyboard_stop = [[Button.inline("Stop and reset conversation", b"stop")]]


# Define helper function to retrieve a message from a conversation and handle button clicks
async def send_question_and_retrieve_result(prompt, conv, keyboard):
    """
    Sends a question to the user and retrieves their response.

    Args:
        prompt (str): The question to ask the user.
        conv (telethon.client.conversations.Conversation): The conversation object to use for sending the message.
        keyboard (list): The keyboard to send with the message.

    Returns:
        Tuple[Union[events.callbackquery.CallbackQuery.Event, str], telethon.types.Message]: A tuple containing the user's response and the message object.
    """
    # Send the prompt with the keyboard to the user and store the sent message object
    message = await conv.send_message(prompt, buttons = keyboard)
    
    # Wait for the user to respond or tap a button using asyncio.wait()
    done, _ = await asyncio.wait({conv.wait_event(events.CallbackQuery()), conv.get_response()}, return_when=asyncio.FIRST_COMPLETED)
    
    # Retrieve the result of the completed coroutine and delete the sent message
    result = done.pop().result()
    await message.delete()
    
    # Return the user's response or None if they tapped a button
    if isinstance(result, events.CallbackQuery.Event):
        return None
    else:
        return result.message.strip()


# Define the main chatbot handler
@client.on(events.NewMessage(pattern="(?i)/start"))
async def handle_start_command(event):
    """
    Starts a new conversation with the user.

    Args:
        event (telethon.events.NewMessage): The event that triggered this function.
    """

    SENDER = event.sender_id
    prompt = "Hello 🤖! I'm Telegram-ChatGPT-Bot, an AI-powered Telegram chatbot ready to assist you with any question you have. Simply ask me anything, and I'll provide you with an answer using advanced language models and machine learning algorithms."
    try:
        # Greet the user
        await client.send_message(SENDER, prompt)

        # Start a conversation
        async with client.conversation(await event.get_chat(), exclusive=True, timeout=600) as conv:
            # Create an empty history to store chat history
            history = []

            # Keep asking for input and generating responses until the conversation times out or the user clicks the stop button
            while True:
                # Prompt the user for input
                prompt = "Please provide your input to Telegram-ChatGPT-Bot"
                user_input = await send_question_and_retrieve_result(prompt, conv, keyboard_stop)
                
                # Check if the user clicked the stop button
                if user_input is None:
                    # If the user clicked the stop button, send a prompt to reset the conversation
                    prompt = "Received. Conversation will be reset. Type /start to start a new one!"
                    await client.send_message(SENDER, prompt)
                    break
                else:
                    # Send a "I'm thinking message..."
                    prompt = "Received! I'm thinking about the response..."
                    thinking_message = await client.send_message(SENDER, prompt)

                    # If the user did not click the stop button, generate a response using OpenAI API
                    # Add the user input to the chat history
                    history.append({"role":"user", "content": user_input})

                    # Generate a chat completion using OpenAI API
                    chat_completion = openai.ChatCompletion.create(
                        model=config.model_engine, # ID of the model to use.
                        messages=history, # The messages to generate chat completions for. This must be a list of dicts!
                        max_tokens=500, # The maximum number of tokens to generate in the completion.
                        n=1, # How many completions to generate for each prompt.
                        temperature=0.1 # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                    )

                    # Retrieve the response from the chat completion
                    response = chat_completion.choices[0].message.content 

                    # Add the response to the chat history
                    history.append({"role": "assistant", "content": response})

                    # Delete the Thinking message
                    await thinking_message.delete()
                    
                    # Send the response to the user
                    await client.send_message(SENDER, response, parse_mode='Markdown')


    except asyncio.TimeoutError:
        # Conversation timed out
        await client.send_message(SENDER, "<b>Conversation ended✔️</b>\nIt's been too long since your last response. Please type /start to begin a new conversation.", parse_mode='html')
        return

    except telethon.errors.common.AlreadyInConversationError:
        # User already in conversation
        pass

    except Exception as e: 
        # Something went wrong
        print(e)
        await client.send_message(SENDER, "<b>Conversation ended✔️</b>\nSomething went wrong. Please type /start to begin a new conversation.", parse_mode='html')
        return


## Main function
if __name__ == "__main__":
    print("Bot Started...")    
    client.run_until_disconnected() # Start the bot!


