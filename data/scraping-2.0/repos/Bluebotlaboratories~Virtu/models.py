import openai
import time
import json
import re
import os

# Handle all the AI Stuff
class OpenAICompletionModel():
    def __init__(self, apiKey):
        self.apiKey = apiKey
        self.defaultApiKey = apiKey

        # Initialise OpenAI
        openai.api_key = self.apiKey
        self.engines = openai.Engine.list()
        ##for engine in self.engines.data:
        ##    if (engine.object == "engine" and engine.ready):
        ##        print(engine.id)
        ##    else:
        ##        pass

        # Initialise "addons"
        self.memory = [
            "Virtu is a large language model trained by OpenAI. It is designed to be a chatbot, and should not autocomplete prompts. Do not autocomplete, complete or edit prompts in any way. knowledge cutoff: 2021-09 Current date: December 10 2022 Browsing: disabled"
        ]

        # Define initialisation prompts
        self.initialisationPrompts = {}
        
        availablePromptFiles = os.listdir("./initialisationPrompts/")
        for promptFile in availablePromptFiles:
            if (promptFile.split('.')[-1] == 'txt'):
                with open(os.path.join("./initialisationPrompts", promptFile), 'r') as file:
                    promptData = file.read()
                    self.initialisationPrompts['.'.join(promptFile.split('.')[:-1])] = promptData.split("===")

        # Define AI Options
        self.defaultConfig = {
            "engine": "text-davinci-003",
            "temperature": 0.5,
            "max_tokens": 512,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "useMemory": True
        }
        self.config = self.defaultConfig

        # Timeout stuff
        self.timeoutReason = ""
        self.timeout = 0

        # "Premium" Stuff
        self.premiumMode = False

    # Premium stuff
    def enablePremiumMode(self, apiKey):
        if (apiKey != self.defaultApiKey):
            try:
                openai.api_key = apiKey
                self.engines = openai.Engine.list()
                self.premiumMode = True
                self.apiKey = apiKey
                return True
            except:
                return False
        else:
            return False

    # Reset memory
    def resetMemory(self):
        self.memory = self.memory[:1]
    
    # Import ChatGPT history
    def importMemory(self, memoryToImport):
        try:
            self.resetMemory()
            currentChatItem = "User: "

            for memoryItem in memoryToImport:
                self.memory.append(currentChatItem + memoryItem)
                if (currentChatItem == "User: "):
                    currentChatItem = "Response: "
                else:
                    currentChatItem = "User: "

            if (currentChatItem != "User: "):
                prompt = self.memory[-1]
                self.memory = self.memory[:-1]
                response = self.processPrompt(prompt)
                response += "\n\n" + "Chat Imported Successfuly"
            else:
                response = "Chat Imported Successfuly"

            return response
        except Exception as e:
            return "Error Importing Chat: " + str(e)

    # Actual AI part
    def processPrompt(self, prompt):
        if (time.time() < self.timeout):
            return "Timed out - Please wait [" + str(round(self.timeout - time.time())) + "] seconds...\nReason:\n" + self.timeoutReason

        # Add prompt to memory
        self.memory.append("User: " + prompt)
        self.memory.append("Response: ")

        # Haha big brain go brrrrrrr
        error = True
        try:
            openai.api_key = self.apiKey

            completion = openai.Completion.create(
                engine=self.config["engine"],
                prompt='\n'.join(self.memory), #prompt
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                top_p=self.config["top_p"],
                frequency_penalty=self.config["frequency_penalty"],
                presence_penalty=self.config["presence_penalty"],
            )
        
            response = completion.choices[0].text
            if (len(response.strip()) >= 9 and response.strip().lower()[:9] == "response: "):
                response = re.sub('response: ', '', response, 1, re.I)
            elif (len(response.strip()) >= 9 and response.strip().lower()[:9] == "response:"):
                response = re.sub('response:', '', response, 1, re.I)
            elif ("response:" in response.lower()):
                response = re.sub('response:', '', response, 1, re.I)

            response = response.lstrip()

            #print("Response: " + response)

            # Add response to memory and return it
            self.memory[-1] = "Response: " + response
            error = False
        except openai.error.RateLimitError as e:
            response = "[RATELIMITED - PLEASE WAIT 30 SECONDS]\n`" + str(e) + "`"
            self.timeout = time.time() + 30
            self.timeoutReason = "OpenAI rate limited"
        except openai.error.InvalidRequestError as e:
            print(e)
            response = "[HISTORY FULL - PLEASE RESET]"

        if (not self.premiumMode and not error):
            self.timeout = time.time() + 30
            self.timeoutReason = "--__Remove timeouts with Virtu Premium:__--\nLiterally just provide your own api key, see more info in `/config Premium Mode`"
        return response

    def processInitialisationPrompt(self, prompt):
        self.resetMemory()
        response = ""

        try:
            with open(os.path.join("./initialisationPrompts", prompt + "_config.json"), 'r') as aiConfigJSON:
                self.config = json.loads(aiConfigJSON.read())
        except:
            self.config = self.defaultConfig

        print("Initialised, Using config:", self.config)

        for promptToInitialiseWith in self.initialisationPrompts[prompt]:
            response = self.processPrompt(promptToInitialiseWith)

        #print('\n'.join(self.memory))

        return response