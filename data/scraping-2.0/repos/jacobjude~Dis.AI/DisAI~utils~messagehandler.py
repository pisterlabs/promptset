import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import ceil

import aiohttp
import discord
import openai
from discord import PartialEmoji

from config import (CLAIM_CREDITS_AMOUNT, ICON_URL, MAX_TOKENS, MEMORY_LENGTH,
                    OPENAI_API_KEY, VOTE_CREDITS_AMOUNT, functions)
from core.Server import Server
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from extensions.constants import Analytics
from extensions.embeds import get_credits_needed_embed
from extensions.helpers import (get_credits_cost, has_time_passed,
                                send_error_message, update_analytics, get_tokens)
from extensions.uiactions import CreditsView
import utils.pineconehandler as pineconehandler

# Constants
CHUNK_SIZE = 1970
LOADING_EMOJI = PartialEmoji(name="", animated=True, id=1120087219928051843)

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Set up OpenAI
openai.api_key = OPENAI_API_KEY
MEMORY_LENGTH = int(MEMORY_LENGTH)
MAX_TOKENS = int(MAX_TOKENS)
VOTE_CREDITS_AMOUNT = int(VOTE_CREDITS_AMOUNT)
CLAIM_CREDITS_AMOUNT = int(CLAIM_CREDITS_AMOUNT)

# Load cookies
try:
    with open("./cookies.json", "r", encoding="utf-8") as f:
        db_handler_cookies = json.load(f)
except FileNotFoundError:
    logger.error("Cookies file not found.")
    db_handler_cookies = {}



async def pre_gpt_response(converse_mode, regen_mode, platform, chatbot, user_message, should_append_context, processed_chatbots):
    """Do things before getting a GPT response."""
    if not converse_mode and not regen_mode: # if it's in converse mode, then don't pamper context / get bing responses, since that's already been done 
        await pamper_context(platform.id, chatbot, user_message, should_append_context)
    platform.last_interaction_date = datetime.now().replace(microsecond=0) # update last interaction date for the platform
    for processed_chatbot in processed_chatbots: # add the messages of the other, processed chatbots to the current chatbot
        for entry in reversed(processed_chatbot.context): # find last assistant message
            if entry['role'] == 'assistant':
                last_assistant_message = entry['content']
                break
        else:
            last_assistant_message = ""
        if last_assistant_message:
            chatbot.context.append({'role': 'user','content': f"{processed_chatbot.name}: {last_assistant_message}"})
    chatbot.context.append({'role': 'assistant','content': ""})

async def handle_gpt_response(platform, chatbot, user_message, credits_cost, response_message=None, converse_mode=False, should_make_buttons=True, 
                              should_append_context=True, regen_mode=False, processed_chatbots=[]):
    try:
        if platform.credits - credits_cost < 0: # if not enough credits, return False
            if has_time_passed(platform.last_creditsembed_date, 45):
                platform.last_creditsembed_date = datetime.now().replace(microsecond=0)
                await update_analytics(platform.analytics, Analytics.RAN_OUT_OF_CREDITS.value)
                await user_message.channel.send(embed=await get_credits_needed_embed(chatbot), view=CreditsView(platform))
            return False  
        try:
            if chatbot.last_message:
                await chatbot.last_message.clear_reactions()
        except:
            pass
        
        response_message = await handle_gpt_output_server(chatbot, user_message, response_message, converse_mode, regen_mode, platform, should_append_context, processed_chatbots)
    except Exception as e:
        logger.error(f"{platform.id} {chatbot.name} - handle gpt response err :{type(e)} - {e}\n{chatbot.context}")
        if "avatar_url" in str(e):
            await send_error_message("The avatar URL you set is invalid. Please set a valid URL from /settings.", user_message)
            chatbot.avatar_url = ICON_URL
        else:
            await send_error_message(f"An unexpected error occurred. Please join the community server and report this bug:\n{type(e)}\n{e}", user_message)
            logger.error(f"{platform.id} {chatbot.name} - handle gpt response err :{type(e)} - {e}\n{chatbot.context}")
        return False
    if not response_message:
        return False
    chatbot.last_message = response_message # is a webhook message
    # add reactions
    if should_make_buttons:
        await response_message.add_reaction('⏩')
        await response_message.add_reaction('🔃')
    if not converse_mode and (should_make_buttons):
        await response_message.add_reaction('🗑️')
    logger.info(f"Finished and got GPT response. Platform: {platform.name} - {platform.id} - Chatbot: {chatbot.name}")
    return True
        
    
async def handle_gpt_output_server(chatbot, user_message, response_message, converse_mode, regen_mode, platform, should_append_context, processed_chatbots):
    logger.info("In handle gpt output server")
    if isinstance(user_message.channel, discord.Thread): # have to change some of the terminology if the message channel is a thread
        channel = user_message.channel.parent
        thread_to_send = user_message.channel
    else:
        thread_to_send = None
        channel = user_message.channel
    webhooks = await channel.webhooks() # get the webhooks. if no DisAI webhook in the channel, create one. we use webhooks to send messages to get custom avatar URLs and names
    og_webhook = None # the original webhook  
    for webhook in webhooks:
        if webhook.name == "Dis.AI Webhook":
            og_webhook = webhook
            break
    else:
        og_webhook = await channel.create_webhook(name="Dis.AI Webhook")
    if response_message: # response_message is a webhook message from dis.ai. if already there, edit it
        await response_message.edit(content=str(LOADING_EMOJI))
    else: # otherwise, gotta send a message
        
        if thread_to_send:
            response_message = await og_webhook.send(content=str(LOADING_EMOJI), username=chatbot.name, avatar_url=chatbot.avatar_url, thread=thread_to_send, wait=True)
        else:
            response_message = await og_webhook.send(content=str(LOADING_EMOJI), username=chatbot.name, avatar_url=chatbot.avatar_url, wait=True)
    tic = time.perf_counter()
    await pre_gpt_response(converse_mode, regen_mode, platform, chatbot, user_message, should_append_context, processed_chatbots)
    toc = time.perf_counter()
    logger.info(f"{platform.name} ({platform.id}) - {chatbot.name}: pre gpt response took {toc-tic} seconds")
    i = 0 # represents the tokens
    numbreaks = 0 # the number of CHUNK_SIZE-character batches
    chunks = [] # store the batches
    context = chatbot.context[-1] # refer to the assistant content we just appended to 
    function_call_details = [] # if there's a function call, store the details here (since we stream it)
    async with aiohttp.ClientSession() as session:
        if chatbot.web_search:
            payload = {
                "model": chatbot.model,
                "messages": chatbot.context,
                "max_tokens": MAX_TOKENS,
                "temperature": chatbot.temperature,
                "top_p": chatbot.top_p,
                "presence_penalty": chatbot.presence_penalty,
                "frequency_penalty": chatbot.frequency_penalty,
                "stream": True,
                "functions": functions,
                "function_call": "auto"
            }
        else:
            payload = {
                "model": chatbot.model,
                "messages": chatbot.context,
                "max_tokens": MAX_TOKENS,
                "temperature": chatbot.temperature,
                "top_p": chatbot.top_p,
                "presence_penalty": chatbot.presence_penalty,
                "frequency_penalty": chatbot.frequency_penalty,
                "stream": True
            }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as resp:
            async for data in resp.content.iter_any(): 
                data_strings = data.decode('utf-8').strip().split("\n\n") # convert the data to a json
                for string in data_strings:
                    if string != "data: [DONE]":
                        data_dict = json.loads(string.replace("data: ", "")) # get rid of the 'data: ' prefix so we can convert it to a dictionary
                    else:
                        break
                    if 'error' in data_dict: # check for errors
                        logger.error(f"handle gpt response ERROR: {platform.name} ({platform.id}) - {chatbot.name}: {data_dict}")
                        del chatbot.context[-2:]
                        if data_dict['error']['code'] == "context_length_exceeded":
                            await send_error_message("The chat history is too long for this GPT Model.\nPlease use `/clearmemory` and try again.", user_message)
                        elif data_dict['error']['type'] == 'server_error':
                            logger.error(f"OpenAI server error: {data_strings}")
                            await send_error_message("OpenAI had a server error. Please try again!\n(Sorry, this is OpenAI's fault, not ours!)", user_message)
                        else:
                            logger.error(f"Unexpected error: {data_strings}")
                            await send_error_message(f"There was a small hiccup getting your response. Please try again.", user_message)
                        return
                    if i == 0:
                        if 'function_call' in data_dict['choices'][0]['delta']: # check if the response is empty
                            function_call_details.append(data_dict['choices'][0]['delta']['function_call']['name'])
                        i += 1
                        continue
                    if 'function_call' in data_dict['choices'][0]['delta']:
                        function_call_details.append(data_dict['choices'][0]['delta']['function_call']['arguments'])
                    if 'content' in data_dict['choices'][0]['delta']:
                        chunks.append(data_dict['choices'][0]['delta']['content'])
                    if (i + 1) % 15 == 0:  # reached chunk threshold, update message with new chunk
                        context['content'] += "".join(chunks)
                        chunks = []
                        if response_message is not None: # check if a response_message exists. if it doesn't, then create one by send_channel_msg. if it does, edit the response_message.
                            # messages are chunked in groups in CHUNK_SIZE characters
                            if len(context['content']) / (numbreaks + 1) <= CHUNK_SIZE: # check if the current message would exceed CHUNK_SIZE characters
                                await response_message.edit(content=f"{context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE]} {str(LOADING_EMOJI)}")
                            else: #if yes, then edit the message w/ the rest of the chunk. then send the rest of the message.
                                await response_message.edit(content=f"{context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE]}")
                                numbreaks += 1
                                response_message = await send_channel_msg_for_webhook(og_webhook, f"--{context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE]}", avatar_url=chatbot.avatar_url, chatbot_name=chatbot.name, should_send_LOADING_EMOJI=True, thread_to_send=thread_to_send)
                                
                        else: # response_message is none, which also means no webhook has been made. create the webhook for the chatbot and send the chunk. 
                            response_message = await send_channel_msg_for_webhook(og_webhook, context['content'], avatar_url=chatbot.avatar_url, chatbot_name=chatbot.name, should_send_LOADING_EMOJI=True, thread_to_send=thread_to_send)
                        await asyncio.sleep(1)
                    i += 1
    
    # check for function call. 
    if function_call_details:
        if "google" in function_call_details[0]:
            function_call_query = ''.join(function_call_details[1:])
            parameters = json.loads(function_call_query)
            search_query = []
            for value in parameters.values():
                search_query.append(value)
            await response_message.edit(content=f"🌐 Searching the web for {search_query} {str(LOADING_EMOJI)}\n\n(Toggle web search in `/settings`)")
            if not user_message.channel.id in chatbot.bing_bots or chatbot.bing_bots[user_message.channel.id] == None:
                chatbot.bing_bots[user_message.channel.id] = await Chatbot.create(cookies=db_handler_cookies) 
            search_results = await search_google_using_natural_language(''.join(search_query), chatbot.bing_bots[user_message.channel.id])
            del chatbot.context[-1]
            chatbot.context.append({'role':"function", 'name': 'search_google_using_natural_language', 'content': search_results})
            return await handle_gpt_output_server(chatbot, user_message, response_message, converse_mode, regen_mode, platform, should_append_context=False, processed_chatbots=processed_chatbots)
        
    # send any remaining tokens that haven't been sent yet
    context['content'] += "".join(chunks)
    if response_message != None: # 
        await response_message.edit(content=context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE])
    elif context['content']: # this happens if number of tokens never reaches 50. create a webhook and then send response_message
        # og_webhook = await user_message.channel.create_webhook(name="Dis.AI Webhook")
        if thread_to_send:
            response_message = await og_webhook.send(username=chatbot.name, wait=True, content=context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE], avatar_url=chatbot.avatar_url, thread=thread_to_send)
        else:
            response_message = await og_webhook.send(username=chatbot.name, wait=True, content=context['content'][numbreaks*CHUNK_SIZE:(numbreaks+1)*CHUNK_SIZE], avatar_url=chatbot.avatar_url)
    else:
        logger.error("Uknown error in handle_gpt_response_server.")
        await send_error_message("Unknown error. Please join the support server for more help.", response_message, user_message.channel)
    token_count = get_tokens(chatbot.model, chatbot.context)
    platform.analytics.append((Analytics.GOT_GPT_RESPONSE.value, datetime.now(), token_count))
    return response_message
                                
def find_order_in_string(string, chatbots):
    found_chatbots = {chatbot: string.find(chatbot.name.lower()) for chatbot in chatbots if string.find(chatbot.name.lower()) != -1}
    ordered_chatbots = sorted(found_chatbots.items(), key=lambda t: t[1])
    return [chatbot for chatbot, _ in ordered_chatbots]

async def handle_chatbot_list(chatbot_list, message, botuser, platform, processed_chatbots, ignore_mention_mode=False):
    for i in range(len(chatbot_list)):
        chatbot = chatbot_list[i]
        if message.channel.id in chatbot.channels: # if chatbot is supposed to respond in this channel
            try:
                if (not chatbot.mention_mode or ignore_mention_mode or (chatbot.mention_mode and (botuser in message.mentions or (message.reference and message.reference.cached_message.author.name == chatbot.name)))):
                    
                        cost = get_credits_cost(chatbot.model)
                        try:
                            await processed_chatbots[-1].last_message.clear_reactions()
                        except:
                            pass
                        response_success = await handle_gpt_response(platform, chatbot, message, cost, None, False, chatbot.should_make_buttons, processed_chatbots=processed_chatbots)
                        if response_success:
                            platform.credits -= cost
                        for processed_chatbot in processed_chatbots:
                            processed_chatbot.context.append({'role': 'user', 'content': f"{chatbot.name}: {chatbot.context[-1]['content']}"})
                        processed_chatbots.append(chatbot)
                elif chatbot.mention_mode:
                    chatbot.context.append({'role': 'user', 'content': f"{message.author.display_name}: {message.content}"})
                    for processed_chatbot in processed_chatbots:
                        for entry in reversed(processed_chatbot.context): # find last assistant message
                            if entry['role'] == 'assistant':
                                last_assistant_message = entry['content']
                                break
                        chatbot.context.append({'role': 'user','content': f"{processed_chatbot.name}: {last_assistant_message}"})
                    processed_chatbots.append(chatbot)
            except Exception as e:
                logger.error(f"handle_chatbot_list err {platform.name} ({platform.id}) - {chatbot.name}: {type(e)} - {e}")

async def process_ai_response(platform, message, botuser):
    chatbot_channels = [chatbot.channels for chatbot in platform.chatbots]
    for channels_list in chatbot_channels:
        if message.channel.id in channels_list:
            break
    else:
        return
    processed_chatbots = []
    tic = time.perf_counter()
    chatbots_order = find_order_in_string(message.content[:700].lower(), platform.chatbots)
    toc = time.perf_counter()
    
    if not chatbots_order:
        chatbots_order = []
    await handle_chatbot_list(chatbots_order, message, botuser, platform, processed_chatbots, ignore_mention_mode=True)
    remaining_chatbots = [chatbot for chatbot in platform.chatbots if chatbot not in chatbots_order]
    await handle_chatbot_list(remaining_chatbots, message, botuser, platform, processed_chatbots, ignore_mention_mode=False)
        
        
async def handle_system_message(chatbot, working_index):
    if not chatbot.context or chatbot.context[working_index]['role'] != 'system':
        chatbot.context.insert(working_index, {'role':'system','content':chatbot.prompt})

async def handle_user_message(chatbot, message, should_append_content, working_index):
    content = message.content
    if should_append_content:
        if chatbot.include_usernames:
            content = f"{message.author.display_name}: {content}"
        chatbot.context.append({'role': 'user', 'content': content})

async def handle_long_term_memory(chatbot, platform_id, message, working_index):
    if chatbot.long_term_memory:
        executor = ThreadPoolExecutor()
        if len(chatbot.context) > MEMORY_LENGTH:
            non_system_index = next((index for index, dict in enumerate(chatbot.context) if dict['role'] != 'system'), 0) # index of first non-system message
            chatbot.batch_number = await pineconehandler.upsert_data(chatbot.context[non_system_index:len(chatbot.context) - 3], 
                                                                     f"{platform_id}-{chatbot.name}", chatbot.batch_number + 1, window=2,stride=1,batch_size=10)
            del chatbot.context[non_system_index:len(chatbot.context) - 3]
        try:
            pinecone_query = await asyncio.get_event_loop().run_in_executor(executor, pineconehandler.search_pinecone, message.content, f"{platform_id}-{chatbot.name}")
            if pinecone_query:
                working_index += 1
                queries = '\n'.join(pinecone_query)
                prompt_pamper = f"<Here are some previous chat messages. If relevant, use the information from these messages in your response>\n{queries}\n</END OF PREVIOUS CHAT MESSAGES>"
                if working_index < len(chatbot.context) and chatbot.context[working_index]['content'].endswith("MESSAGES>") and chatbot.context[working_index]['role'] == "system":
                    chatbot.context[working_index]['content'] = prompt_pamper
                else:
                    chatbot.context.insert(working_index, {'role':'system','content':prompt_pamper})
        except Exception as e:
            logger.error(f"pinecone query err: {e}")

async def handle_data(chatbot, platform_id, message, working_index):
    if chatbot.data_name:
        working_index += 1
        executor = ThreadPoolExecutor()
        pinecone_query = await asyncio.get_event_loop().run_in_executor(executor, pineconehandler.search_pinecone, message.content, f"{platform_id}-{chatbot.name}-data")
        if pinecone_query:
            queries = ''.join(pinecone_query)
            if chatbot.data_name[0] == "Y":
                stamp_str = "timestamp"
            else:
                stamp_str = "page number"
            prompt_pamper = f"The user uploaded the data of a {chatbot.data_name}. If relevant, respond to the user using this new information. If you cannot answer the user, tell them to be more specific about the question. Always include the {stamp_str} from which you derived your answer. The following is an excerpt from the new information.\n<Excerpt>\n{queries}\n</Excerpt>"
            if working_index < len(chatbot.context) and chatbot.context[working_index]['content'].endswith("</Excerpt>") and chatbot.context[0]['role'] == "system":
                chatbot.context[working_index]['content'] = prompt_pamper.replace('\n', '')
            else:
                chatbot.context.insert(working_index, {'role':'system','content':prompt_pamper})


async def handle_lorebooks(chatbot, platform_id, message, working_index):
    executor = ThreadPoolExecutor()
    for lorebook in chatbot.lorebooks:
        working_index += 1
        pinecone_query = await asyncio.get_event_loop().run_in_executor(executor, pineconehandler.search_pinecone, message.content, f"{platform_id}-{chatbot.name}-{lorebook}")
        if pinecone_query:
            queries = ''.join(pinecone_query)
            prompt_pamper = f"[Important character and world information for {chatbot.name}]:\n{queries}\n[End of character and world information]. Be sure to dynamically and creatively use this information in your response."
            if working_index < len(chatbot.context) and chatbot.context[working_index]['content'].endswith("r response.") and chatbot.context[0]['role'] == "system":
                chatbot.context[working_index]['content'] = prompt_pamper.replace('\n', '')
            else:
                chatbot.context.insert(working_index, {'role':'system','content':prompt_pamper})

async def pamper_context(platform_id, chatbot, message, should_append_content=True):
    try:
        working_index = 0
        await handle_system_message(chatbot, working_index)
        await handle_lorebooks(chatbot, platform_id, message, working_index)
        await handle_data(chatbot, platform_id, message, working_index)
        await handle_long_term_memory(chatbot, platform_id, message, working_index)
        await handle_user_message(chatbot, message, should_append_content, working_index)
    except Exception as e:
        logger.error(f"{platform_id} {chatbot.name} - pamper err: {e}") 
    

async def search_google_using_natural_language(question, bingbot):
    """ Uses bing response, not google. Google name is for prompt purposes only"""
    try:
        result = await bingbot.ask(prompt=question, conversation_style=ConversationStyle.precise, simplify_response=True, locale="en")
        text = f"{result['text']}\n{result['sources_text']}"
        return text
    except Exception as e:
        logger.error(f"bing response err: {type(e)} - {e}")
        await bingbot.close()
        bingbot = None
        return "An error occurred while searching the web. Tell the user to try again or join the Dis.AI support server."
        

async def get_moderation(question):
    try:
        response = openai.Moderation.create(input=question)
    except Exception as e:
        return  -2
    if response.results[0].flagged:
        return -1
    return -3

async def send_channel_msg_for_webhook(channel, msg, view=None, delete_after=None, avatar_url="", chatbot_name="Dis.AI", should_send_LOADING_EMOJI=False, thread_to_send=None):
    """if it aint broke don't fix it"""
    chunk_size = 1970
    max_msgs = ceil(len(msg) / chunk_size)
    response_channel = None
    if max_msgs > 1:
        for i in range(max_msgs):
            chunk = msg[i*chunk_size:(i+1)*chunk_size]
            if not view:
                
                response_channel = await channel.send(username=chatbot_name, wait=True, content=f"{chunk} ({i+1}/{max_msgs})", avatar_url=avatar_url)
            else:
                response_channel = await channel.send(username=chatbot_name,wait=True, view=view, content=f"{chunk} ({i+1}/{max_msgs})", avatar_url=avatar_url)
        return response_channel
    else:
        content = f"{str(msg)} {str(LOADING_EMOJI)}" if should_send_LOADING_EMOJI else str(msg)
        if view:
            if thread_to_send:
                response_channel = await channel.send(username=chatbot_name,wait=True, view=view, content=content, avatar_url=avatar_url, thread=thread_to_send)
            else:
                response_channel = await channel.send(username=chatbot_name,wait=True, content=content, avatar_url=avatar_url)
        else:
            if thread_to_send:
                response_channel = await channel.send(username=chatbot_name,wait=True, content=content, avatar_url=avatar_url, thread=thread_to_send)
            else:
                response_channel = await channel.send(username=chatbot_name,wait=True, content=content, avatar_url=avatar_url)
        return response_channel
        
async def send_channel_msg(channel, msg, view=None, delete_after=None, should_send_LOADING_EMOJI=False):
    """chunks messages if it's greater than discord's character limit"""
    max_msgs = ceil(len(msg) / 2000)
    response_channel = None
    if max_msgs > 1:
        chunk_size = 1990
        for i in range(max_msgs):
            chunk = msg[i*chunk_size:(i+1)*chunk_size]
            response_channel = await channel.send(view=view, content=f"{chunk} ({i+1}/{max_msgs})", delete_after=delete_after)
        return response_channel
    else:
        content = f"{str(msg)} {str(LOADING_EMOJI)}" if should_send_LOADING_EMOJI else str(msg)
        response_channel = await channel.send(view=view, content=content, delete_after=delete_after)
        return response_channel
    
async def send_channel_msg_as_embed(interaction, msg, title, delete_after=None):
    """does the same thing as above but with embeds"""
    try:
        chunk_size = 4095
        max_msgs = ceil(len(msg) / chunk_size)
        
        for i in range(max_msgs):
            chunk = msg[i * chunk_size : (i + 1) * chunk_size]
            embed = discord.Embed(title=title, description=chunk, color=discord.Colour.blue())
            if max_msgs > 1:
                embed.set_footer(text=f"{i+1}/{max_msgs}")
            await interaction.channel.send(embed=embed, delete_after=delete_after)
    except Exception as e:
        logger.error(f"send_channel_msg_as_embed err: {e}")

    
    
        
