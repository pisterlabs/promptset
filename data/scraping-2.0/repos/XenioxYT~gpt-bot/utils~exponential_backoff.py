import requests
import json
import re
import time
import itertools
import openai
import sqlite3
import discord
import aiohttp
import asyncio

# If anyone is reading this, I'm sorry for the mess. I'm not a good programmer. This code is so all over the place it's not even funny 
# (I made it and even I can't understand it anymore).

def get_latest_conversation(conversation_id):
    db_conn = sqlite3.connect('conversations.db')
    cursor = db_conn.cursor()
    cursor.execute("SELECT conversation FROM conversations WHERE conversation_id = ?", (conversation_id,))
    result = cursor.fetchone()
    db_conn.close()
    if result:
        return json.loads(result[0])
    return None

async def exponential_backoff(api_call, conversation_id, message, max_retries=5):
    models = ["gpt-4-1106-preview", "gpt-4-1106-preview"]
    model_cycle = itertools.cycle(models)
    
    delay = 1
    
    for i in range(max_retries):
        model = next(model_cycle)
        print(f"\033[32mUsing model {model}.\033[0m")
        
        latest_conversation = get_latest_conversation(conversation_id)
        # print(latest_conversation)
        
        try:
            return await asyncio.wait_for(api_call(model, latest_conversation), timeout=10)
        except asyncio.TimeoutError:
            print(f"API call timed out after 10 seconds using model {model}.")
            continue
        except (requests.RequestException, Exception, openai.OpenAIError, openai.InvalidRequestError) as e:
            error_msg = str(e)
            print(f"Error message: {error_msg}")

            if "Rate limit reached" in error_msg:
                wait_time = int(re.findall(r"\d+", error_msg)[0])
                print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif "flagged moderation category:" in error_msg:
                category = re.search(r"flagged moderation category: (.+?)$", error_msg).group(1)
                print(f"Flagged moderation category: {category}. Retrying with next model...")
                modify_conversation(category, conversation_id)
                print("Conversation modified. Removed last few messages from the user.")

                if message.guild:  # Check if the message is in a server
                    await message.delete()
                    await message.channel.send(f"Your message was ignored because of moderation category: {category}. Please be more respectful in the future. The last few messages in the conversation were removed.")
                else:  # If it's in a DM
                    await message.channel.send(f"Your message was ignored because of moderation category: {category}. Please be more respectful in the future. The last few messages in the conversation were removed.")
                break
            
            print(f"Error occurred: {e}. Retrying in {delay} seconds...")
            print(f"\033[31m{model} didn't work on attempt {i+1}...\033[0m")
            await asyncio.sleep(delay)
            delay *= 2
    
    response = "I'm really sorry but I'm having some trouble connecting with the service I need to chat properly. Could you please reach out to Xeniox if this keeps happening?"
    await message.delete()
    await message.channel.send(response)
    raise Exception("API call failed after maximum number of retries with all models")

def modify_conversation(category, conversation_id):
    db_conn = sqlite3.connect('conversations.db')
    cursor = db_conn.cursor()
    
    # Select conversation using the conversation_id
    cursor.execute("SELECT conversation FROM conversations WHERE conversation_id = ?", (conversation_id,))
    result = cursor.fetchone()
    
    if result:
        conversation = json.loads(result[0])
        
        if conversation and len(conversation) >= 3:  # Check if there are at least 2 messages
            conversation[-1]['content'] = f'[flagged for moderation category: {category}]'
            conversation[-2]['content'] = f'[]'
            conversation[-3]['content'] = f'[]'
            
            cursor.execute("UPDATE conversations SET conversation = ? WHERE conversation_id = ?", (json.dumps(conversation), conversation_id))
            db_conn.commit()
    
    db_conn.close()