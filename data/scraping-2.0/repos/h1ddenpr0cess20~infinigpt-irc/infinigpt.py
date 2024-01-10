'''
InfiniGPT-IRC
    An OpenAI GPT-3.5-Turbo chatbot for internet relay chat with infinite personalities
    written by Dustin Whyte
    April 2023

'''

import irc.bot
from openai import OpenAI
import time
import textwrap
import threading

class ircGPT(irc.bot.SingleServerIRCBot):
    def __init__(self, api_key, personality, channel, nickname, server, password=None, port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.client = OpenAI(api_key=api_key)
        
        self.personality = personality
        self.channel = channel
        self.server = server
        self.nickname = nickname
        self.password = password

        self.messages = {} #Holds chat history
        self.users = [] #List of users in the channel

        #set model, change to gpt-4-1106-preview if you want to use gpt-4-turbo
        self.model = 'gpt-3.5-turbo-1106'

        # prompt parts (this prompt was engineered by me and works almost always)
        self.prompt = ("assume the personality of ", ".  roleplay as them and never break character unless asked.  keep your responses relatively short.")
    
    def chop(self, message):
        lines = message.splitlines()
        newlines = []  # Initialize an empty list to store wrapped lines

        for line in lines:
            if len(line) > 420:
                wrapped_lines = textwrap.wrap(line,
                                            width=420,
                                            drop_whitespace=False,
                                            replace_whitespace=False,
                                            fix_sentence_endings=True,
                                            break_long_words=False)
                newlines.extend(wrapped_lines)  # Extend the list with wrapped lines
            else:
                newlines.append(line)  # Add the original line to the list

        return newlines  # Return the list of wrapped lines
    
    #resets bot to preset personality per user    
    def reset(self, sender):
        if sender in self.messages:
            self.messages[sender].clear()
            self.persona(self.personality, sender)

    #sets the bot personality 
    def persona(self, persona, sender):
        #clear existing history
        if sender in self.messages:
            self.messages[sender].clear()
        personality = self.prompt[0] + persona + self.prompt[1]
        self.add_history("system", sender, personality)

    #set a custom prompt (such as one from awesome-chatgpt-prompts)
    def custom(self, prompt, sender):
        #clear existing history
        if sender in self.messages:
            self.messages[sender].clear()
        self.add_history("system", sender, prompt)

    #adds messages to self.messages    
    def add_history(self, role, sender, message):
        if sender in self.messages:
            self.messages[sender].append({"role": role, "content": message})
        else:
            if role == "system":
                self.messages[sender] = [{"role": role, "content": message}]
            else:
                self.messages[sender] = [
                    {"role": "system", "content": self.prompt[0] + self.personality + self.prompt[1]},
                    {"role": role, "content": message}]

    #respond with GPT model           
    def respond(self, c, sender, message, sender2=None):
        try:
            response = self.client.chat.completions.create(model=self.model, messages=self.messages[sender])
            response_text = response.choices[0].message.content
            
            #removes any unwanted quotation marks from responses
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text.strip('"')

            #add the response text to the history before breaking it up
            self.add_history("assistant", sender, response_text)

            #add username before response
            #if .x function used
            if sender2:
                c.privmsg(self.channel, sender2 + ":")
            #normal .ai usage
            else:
                c.privmsg(self.channel, sender + ":")
            time.sleep(1)

            #split up the response to fit irc length limit
            lines = self.chop(response_text)
            for line in lines:
                c.privmsg(self.channel, line)
                time.sleep(2)

        except Exception as x: #improve this later with specific errors (token error, invalid request error etc)
            c.privmsg(self.channel, "Something went wrong, try again.")
            print(x)

        #trim history for token size management
        if len(self.messages[sender]) > 20:
            del self.messages[sender][1:3]
        
    #run message through moderation endpoint for ToS check        
    def moderate(self, message):
        flagged = False
        if not flagged:
            try:
                moderate = self.client.moderations.create(input=message,) #run through the moderation endpoint
                flagged = moderate.results[0].flagged #true or false
            except:
                pass
        return flagged
              
    #when bot joins network, identify and wait, then join channel   
    def on_welcome(self, c, e):
        #if nick has a password
        if self.password != None:
          c.privmsg("NickServ", f"IDENTIFY {self.password}")
          #wait for identify to finish
          time.sleep(5)
        
        #join channel
        c.join(self.channel)
              
        # get users in channel
        c.send_raw("NAMES " + self.channel)

        #optional join message
        greet = "introduce yourself"
        try:
            response = self.client.chat.completions.create(model=self.model, 
                    messages=[{"role": "system", "content": self.prompt[0] + self.personality + self.prompt[1]},
                    {"role": "user", "content": greet}])
            response_text = response.choices[0].message.content
            lines = self.chop(response_text + f"  Type .help {self.nickname} to learn how to use me.")
            for line in lines:
                c.privmsg(self.channel, line)
                time.sleep(2)
        except:
            pass
            
    def on_nicknameinuse(self, c, e):
        #add an underscore if nickname is in use
        c.nick(c.get_nickname() + "_")

    # actions to take when a user joins 
    def on_join(self, c, e):
        user = e.source
        user = user.split("!")
        user = user[0]
        if user not in self.users:
            self.users.append(user)

    # Optional greeting for when a user joins        
        # greet = f"come up with a unique greeting for the user {user}"
        # if user != self.nickname:
        #     try:
        #         response = self.client.chat.completions.create(model=self.model, 
        #                 messages=[{"role": "system", "content": self.prompt[0] + self.personality + self.prompt[1]}, {"role": "user", "content": greet}])
        #         response_text = response.choices[0].message.content
        #         time.sleep(5)
        #         lines = self.chop(response_text)
        #         for line in lines:
        #             c.privmsg(self.channel, line)
        #             time.sleep(2)
        #     except:
        #         pass
            
    # Get the users in the channel
    def on_namreply(self, c, e):
        symbols = {"@", "+", "%", "&", "~"} #symbols for ops and voiced
        userlist = e.arguments[2].split()
        for name in userlist:
            for symbol in symbols:
                if name.startswith(symbol):
                    name = name.lstrip(symbol)
            if name not in self.users:
                self.users.append(name)       

    #process chat messages
    def on_pubmsg(self, c, e):
        #message parts
        message = e.arguments[0]
        sender = e.source
        sender = sender.split("!")
        sender = sender[0]

        #if the bot didn't send the message
        if sender != self.nickname:
            #basic use
            if message.startswith(".ai") or message.startswith(self.nickname):
                m = message.split(" ", 1)
                m = m[1]

                #moderation   
                flagged = self.moderate(m)  #set to False if you want to bypass moderation
                if flagged:
                    c.privmsg(self.channel, f"{sender}: This message violates OpenAI terms of use and was not sent")
                    
                    #add way to ignore user after a certain number of violations 
                    #maybe like 3 flagged messages gets you ignored for a while

                else:
                    #add to history and start respond thread
                    self.add_history("user", sender, m)
                    thread = threading.Thread(target=self.respond, args=(c, sender, self.messages[sender]))
                    thread.start()
                    thread.join(timeout=30)
                    time.sleep(2) #help prevent mixing user output

            #collborative use
            if message.startswith(".x "):
                m = message.split(" ", 2)
                m.pop(0)
                if len(m) > 1:
                    #get users in channel
                    c.send_raw("NAMES " + self.channel)

                    #check if the message starts with a name in the history
                    for name in self.users:
                        if type(name) == str and m[0] == name:
                            user = m[0]
                            m = m[1]
                            
                            #if so, respond, otherwise ignore
                            if user in self.messages:
                                flagged = self.moderate(m)  #set to False if you want to bypass moderation
                                if flagged:
                                    c.privmsg(self.channel, f"{sender}: This message violates OpenAI terms of use and was not sent")
                                    #add way to ignore user after a certain number of violations

                                else:
                                    self.add_history("user", user, m)
                                    thread = threading.Thread(target=self.respond, args=(c, user, self.messages[user],), kwargs={'sender2': sender})
                                    thread.start()
                                    thread.join(timeout=30)
                                    time.sleep(2)
                            
            #change personality    
            if message.startswith(".persona "):
                m = message.split(" ", 1)
                m = m[1]
                #check if it violates ToS
                flagged = self.moderate(m) #set to False if you want to bypass moderation
                if flagged:
                    c.privmsg(self.channel, f"{sender}: This persona violates OpenAI terms of use and was not set.")
                    #add way to ignore user after a certain number of violations
                else:
                    self.persona(m, sender)
                    thread = threading.Thread(target=self.respond, args=(c, sender, self.messages[sender]))
                    thread.start()
                    thread.join(timeout=30)
                    time.sleep(2)

            #use custom prompts 
            if message.startswith(".custom "):
                m = message.split(" ", 1)
                m = m[1]
                #check if it violates ToS
                flagged = self.moderate(m) #set to False if you want to bypass moderation
                if flagged:
                    c.privmsg(self.channel, f"{sender}: This custom prompt violates OpenAI terms of use and was not set.")
                    #add way to ignore user after a certain number of violations
                else:
                    self.custom(m, sender)
                    thread = threading.Thread(target=self.respond, args=(c, sender, self.messages[sender]))
                    thread.start()
                    thread.join(timeout=30)
                    time.sleep(2)
                    
            #reset to default personality    
            if message.startswith(".reset"):
                self.reset(sender)
                c.privmsg(self.channel, f"{self.nickname} reset to default for {sender}.")

            #stock GPT settings    
            if message.startswith(".stock"):
                if sender in self.messages:
                    self.messages[sender].clear()
                else:
                    self.messages[sender] = []                    
                c.privmsg(self.channel, f"Stock settings applied for {sender}")

            #help menu    
            if message.startswith(f".help {self.nickname}"):
                help = [
                    "I am an OpenAI chatbot.  I can have any personality you want me to have.  Each user has their own chat history and personality setting.",
                    f".ai <message> or {self.nickname}: <message> to talk to me.", ".x <user> <message> to talk to another user's history for collaboration.",
                    ".persona <personality> to change my personality. I can be any personality type, character, inanimate object, place, concept.", 
                    ".custom <prompt> to use a custom prompt instead of a persona",
                    ".stock to set to stock GPT settings.", f".reset to reset to my default personality, {self.personality}.",  
                    "Available at https://github.com/h1ddenpr0cess20/infinigpt-irc"

                ]
                for line in help:
                    c.notice(sender, line)
                    time.sleep(1)
                
if __name__ == "__main__":

    # Set up the OpenAI API client
    api_key = "API_KEY"

    # create the bot and connect to the server
    personality = "an AI that can assume any personality, named InfiniGPT"  #you can put anything here.  A character, person, personality type, object, concept, emoji, etc
    channel = "#CHANNEL"
    nickname = "NICKNAME"
    #password = "PASSWORD"
    server = "SERVER"
    
    #checks if password variable exists (comment it out if unregistered)
    try:
      infiniGPT = ircGPT(api_key, personality, channel, nickname, server, password)
    except:
      infiniGPT = ircGPT(api_key, personality, channel, nickname, server)
      
    infiniGPT.start()

