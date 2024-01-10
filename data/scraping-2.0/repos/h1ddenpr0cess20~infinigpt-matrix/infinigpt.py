"""
infiniGPT: An OpenAI chatbot for the Matrix chat protocol with infinite personalities.

Author: Dustin Whyte
Date: May 2023
"""

import asyncio
from nio import AsyncClient, MatrixRoom, RoomMessageText
import datetime
from openai import OpenAI
import os

class InfiniGPT:
    def __init__(self, server, username, password, channels, personality, api_key):
        self.server = server
        self.username = username
        self.password = password
        self.channels = channels
        self.personality =  personality
        
        self.client = AsyncClient(server, username)
        self.openai = OpenAI(api_key=api_key)
        
        # time program started and joined channels
        self.join_time = datetime.datetime.now()
        
        # store chat history
        self.messages = {}

        #prompt parts
        self.prompt = ("assume the personality of ", ".  roleplay and never break character. keep your responses relatively short.")
        
        #set GPT model 
        #default setting is gpt-3.5-turbo for pricing reasons
        #change to gpt-4-1106-preview if you want to use gpt-4-turbo
        self.model = "gpt-3.5-turbo-1106"
    
    # get the display name for a user
    async def display_name(self, user):
        try:
            name = await self.client.get_displayname(user)
            return name.displayname
        except Exception as e:
            print(e)

    # simplifies sending messages to the channel            
    async def send_message(self, channel, message):
         await self.client.room_send(
            room_id=channel,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message},
        )

    # run message through moderation endpoint
    async def moderate(self, message):
        flagged = False
        if not flagged:
            try:
                moderate = self.openai.moderations.create(input=message,) #run through the moderation endpoint
                flagged = moderate.results[0].flagged #true or false
            except:
                pass
        return flagged

    # add messages to the history dictionary
    async def add_history(self, role, channel, sender, message):
        
        #check if channel is in the history yet
        if channel in self.messages:
            #check if user is in channel history
            if sender in self.messages[channel]: 
                self.messages[channel][sender].append({"role": role, "content": message}) #add the message
            else:
                self.messages[channel][sender] = [
                    {"role": "system", "content": self.prompt[0] + self.personality + self.prompt[1]},
                    {"role": role, "content": message}]
        else:
            #set up channel in history
            self.messages[channel]= {}
            self.messages[channel][sender] = {}
            if role == "system":
                self.messages[channel][sender] = [{"role": role, "content": message}]
            else: 
                #add personality to the new user entry
                self.messages[channel][sender] = [
                    {"role": "system", "content": self.prompt[0] + self.personality + self.prompt[1]},
                    {"role": role, "content": message}]

    # create GPT response
    async def respond(self, channel, sender, message, sender2=None):
        
        try:
            #Generate response with openai model
            response = self.openai.chat.completions.create(
                    model=self.model,
                    temperature=1,
                    messages=message)    
        except Exception as e:
            await self.send_message(channel, "Something went wrong")
            print(e)
        else:
            #Extract response text
            response_text = response.choices[0].message.content
            
            #check for unwanted quotation marks around response and remove them
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text.strip('"')

            #add to history
            await self.add_history("assistant", channel, sender, response_text)
            # .x function was used
            if sender2:
                display_name = await self.display_name(sender2)
            # .ai was used
            else:
                display_name = await self.display_name(sender)
            response_text = display_name + ":\n" + response_text.strip()
            #Send response to channel
            try:
                await self.send_message(channel, response_text)
            except Exception as e: 
                print(e)
            #Shrink history list for token size management (also prevents rate limit error)
            if len(self.messages[channel][sender]) > 24:
                del self.messages[channel][sender][1:3]  #delete the first set of question and answers 

    # change the personality of the bot
    async def persona(self, channel, sender, persona):
        #clear existing history
        try:
            await self.messages[channel][sender].clear()
        except:
            pass
        personality = self.prompt[0] + persona + self.prompt[1]
        #set system prompt
        await self.add_history("system", channel, sender, personality)
        
    # use a custom prompt from other sources like awesome-chatgpt-prompts
    async def custom(self, channel, sender, prompt):
        try:
            await self.messages[channel][sender].clear()
        except:
            pass
        await self.add_history("system", channel, sender, prompt) 
      
    # tracks the messages in channels
    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
       
        # Main bot functionality
        if isinstance(event, RoomMessageText):
            # convert timestamp
            message_time = event.server_timestamp / 1000
            message_time = datetime.datetime.fromtimestamp(message_time)
            # assign parts of event to variables
            message = event.body
            sender = event.sender
            sender_display = await self.display_name(sender)
            room_id = room.room_id
            user = await self.display_name(event.sender)

            #check if the message was sent after joining and not by the bot
            if message_time > self.join_time and sender != self.username:

                # main AI response functionality
                if message.startswith(".ai ") or message.startswith(self.bot_id):
                    m = message.split(" ", 1)
                    m = m[1]
                    # check if it violates ToS
                    flagged = await self.moderate(m)
                    if flagged:
                        await self.send_message(room_id, f"{sender_display}: This message violates the OpenAI usage policy and was not sent.")
                        #add a way to penalize repeated violations here, maybe ignore for x amount of time after three violations

                    else:
                        await self.add_history("user", room_id, sender, m)
                        await self.respond(room_id, sender, self.messages[room_id][sender])

                # collaborative functionality
                if message.startswith(".x "):
                    m = message.split(" ", 2)
                    m.pop(0)
                    if len(m) > 1:
                        disp_name = m[0]
                        name_id = ""
                        m = m[1]
                        if room_id in self.messages:
                            for user in self.messages[room_id]:
                                try:
                                    username = await self.display_name(user)
                                    if disp_name == username:
                                        name_id = user
                                except:
                                    name_id = disp_name
                            flagged = await self.moderate(m)
                            if flagged:
                                await self.send_message(room_id, f"{sender_display}: This message violates the OpenAI usage policy and was not sent.")
                            else:
                                await self.add_history("user", room_id, name_id, m)
                                await self.respond(room_id, name_id, self.messages[room_id][name_id], sender)

                #change personality    
                if message.startswith(".persona "):
                    m = message.split(" ", 1)
                    m = m[1]
                    flagged = await self.moderate(m)
                    if flagged:
                            await self.send_message(room_id, f"{sender_display}: This persona violates the OpenAI usage policy and was not set.  Choose a new persona.")
                    else:
                        await self.persona(room_id, sender, m)
                        await self.respond(room_id, sender, self.messages[room_id][sender])

                #custom prompt use   
                if message.startswith(".custom "):
                    m = message.split(" ", 1)
                    m = m[1]
                    flagged = await self.moderate(m)
                    if flagged:
                            await self.send_message(room_id, f"{sender_display}: This custom prompt violates the OpenAI usage policy and was not set.")
                    else:
                        await self.custom(room_id, sender, m)
                        await self.respond(room_id, sender, self.messages[room_id][sender])
                
                # Secret functions
                if message.startswith(".secret "):
                    secret = {
        'terminal': 'I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. \
                    I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do \
                    not type commands unless I instruct you to do so. When I need to tell you something in English, I will do so by putting text inside \
                    curly brackets {like this}. My first command is pwd',
                    
        'python': 'I want you to act like a Python interpreter. I will give you Python code, and you will execute it. Do not \
                    provide any explanations. Do not respond with anything except the output of the code. The first code is: print("Enter your code")',

        'text game': 'I want you to act as a text based adventure game. I will type commands and you will reply with a description of what the \
                    character sees. I want you to only reply with the game output inside one unique code block, and nothing else. do not write explanations.\
                    do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text \
                    inside curly brackets {like this}. my first command is wake up',
              }
                    m = message.split(" ", 1)
                    m = m[1]
                    
                    if m in secret:
                        if room_id in self.messages:
                            if sender in self.messages[room_id]:
                                self.messages[room_id][sender].clear()
                            else:
                                self.messages[room_id][sender] = {}
                        
                        await self.add_history("system", room_id, sender, secret[m])
                        await self.respond(room_id, sender, self.messages[room_id][sender])
                
                # reset bot to default personality
                if message.startswith(".reset"):
                    if room_id in self.messages:
                        if sender in self.messages[room_id]:
                            self.messages[room_id][sender].clear()
                            await self.persona(room_id, sender, self.personality)
                    try:
                        await self.send_message(room_id, f"{self.bot_id} reset to default for {sender_display}")
                    except:
                        await self.send_message(room_id, f"{self.bot_id} reset to default for {sender}")

                # Stock settings, no personality        
                if message.startswith(".stock"):
                    if room_id in self.messages:
                        if sender in self.messages[room_id]:
                            self.messages[room_id][sender].clear()
                    else:
                        self.messages[room_id] = {}
                        self.messages[room_id][sender] = []
                    try:
                        await self.send_message(room_id, f"Stock settings applied for {sender_display}")
                    except:
                        await self.send_message(room_id, f"Stock settings applied for {sender}")
                
                # help menu
                if message.startswith(".help"):
                    await self.send_message(room_id, 
f'''{self.bot_id}, an OpenAI chatbot.

.ai <message> or {self.bot_id}: <message>
    Basic usage.
    Personality is preset by bot operator.
    This bot is {self.personality}.

.x <user> <message>
    This allows you to talk to another user's chat history.
    <user> is the display name of the user whose history you want to use
    
.persona <personality type or character or inanimate object>
    Changes the personality.  It can be a character, personality type, object, idea.

.custom <custom prompt>
    Allows use of a custom prompt instead of the built-in one

.reset
    Reset to preset personality
    
.stock
    Remove personality and reset to standard GPT settings

Available at https://github.com/h1ddenpr0cess20/infinigpt-matrix    
''')

    # main loop
    async def main(self):
        # Login, print "Logged in as @alice:example.org device id: RANDOMDID"
        print(await self.client.login(self.password))

        # get account display name
        self.bot_id = await self.display_name(self.username)
        
        # join channels
        for channel in self.channels:
            try:
                await self.client.join(channel)
                print(f"{self.bot_id} joined {channel}")
                
            except:
                print(f"Couldn't join {channel}")
        
        # start listening for messages
        self.client.add_event_callback(self.message_callback, RoomMessageText)
                     
        await self.client.sync_forever(timeout=30000)  # milliseconds


if __name__ == "__main__":
    #put a key here and uncomment if not already set in environment
    #os.environ['OPENAI_API_KEY'] = "api_key"

    api_key = os.environ.get("OPENAI_API_KEY")
    
    server = "https://matrix.org" #change if using different homeserver
    username = "@USERNAME:SERVER.TLD" 
    password = "PASSWORD"

    channels = ["#channel1:SERVER.TLD", 
                "#channel2:SERVER.TLD", 
                "#channel3:SERVER.TLD", 
                "!ExAmPleOfApRivAtErOoM:SERVER.TLD", ] #enter the channels you want it to join here
    
    personality = "an AI that can assume any personality, named InfiniGPT" #change to whatever suits your needs
    
    # create bot instance
    infinigpt = InfiniGPT(server, username, password, channels, personality, api_key)
    
    # run main function loop
    asyncio.get_event_loop().run_until_complete(infinigpt.main())

