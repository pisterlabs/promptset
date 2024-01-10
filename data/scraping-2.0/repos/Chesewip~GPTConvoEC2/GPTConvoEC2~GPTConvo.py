
import os
import openai
import random
import re
from ScriptBuilder import *


class GPTConvo:

    def __init__(self, apiKey ):
        self.scriptBuilder = ScriptBuilder();
        self.lastScriptSummary = ""
        self.currentConvoLength = 0;
        openai.api_key = apiKey;
        self.conversation_history = [
        {
            "role": "system",
            "content": self.scriptBuilder.getSystemPrompt()         
        }]

    def createPlot(self, retries = 3):
        if retries < 0:
            print("Failed after several retries.")
            return None
        try:
            self.conversation_history = [
            {
                "role": "system",
                "content": """You are writing a episode plot a funny and adult cartoon.
                                The characters do not move. They just talk about things.
                                Do no create characters. Do not create town names. Do not create any names regarding the show.
                                You can ONLY use these characters """ + self.scriptBuilder.getCharactersString()
            }]
            self.conversation_history.append({"role": "user", "content": self.scriptBuilder.getPlotScript()})
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # use your model
            messages=self.conversation_history
            )
            
            self.conversation_history = [
            {
                "role": "system",
                "content": self.scriptBuilder.getSystemPrompt()         
            }]
            self.conversation_history.append({"role": "user", "content": response['choices'][0]['message']['content']})
            print(response['choices'][0]['message']['content'])
            self.scriptBuilder.plot = self.parse_story(response['choices'][0]['message']['content']);
            return response['choices'][0]['message']['content']

        except Exception as ex:
            print(ex)
            self.callGPT(retries -1)


    def callGPT(self, retries = 3):
        if retries < 0:
            print("Failed after several retries.")
            return None

        if self.currentConvoLength == 0:
            self.conversation_history.append({"role": "user", "content": self.scriptBuilder.getNewScript()})
        else:
            self.conversation_history.append({"role": "user", "content": self.scriptBuilder.getNextScript(self.currentConvoLength)})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-instruct-0914",  # use your model
                messages=self.conversation_history
            )
            #self.conversation_history[1] = ({"role": "assistant", "content": response['choices'][0]['message']['content']})
            self.currentConvoLength += 1
            if self.currentConvoLength > 2:
                self.currentConvoLength = 0;
                self.conversation_history = [
                {
                    "role": "system",
                    "content": self.scriptBuilder.getSystemPrompt()         
                }]
            else:
                self.conversation_history.append({"role": "user", "content": response['choices'][0]['message']['content']})

            return response['choices'][0]['message']['content']

        except Exception as ex:
            print(ex)
            self.callGPT(retries -1)


    def callGPTForOneOffScript(self, retries = 3):
            if retries < 0:
                print("Failed after several retries.")
                return None

            if len(self.conversation_history) > 0:
                self.conversation_history = []

            if len(self.conversation_history) == 0:
                self.conversation_history = [
                {
                    "role": "system",
                    "content": self.scriptBuilder.getSystemPrompt()         
                }]

            self.scriptBuilder = ScriptBuilder();
            self.conversation_history.append({"role": "user", "content": self.scriptBuilder.getNewScript()})

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # use your model
                    messages=self.conversation_history
                )

                self.conversation_history = [
                {
                    "role": "system",
                    "content": self.scriptBuilder.getSystemPrompt()         
                }]

                return response['choices'][0]['message']['content'], self.scriptBuilder.currentDono

            except Exception as ex:
                print(ex)
                self.callGPT(retries -1)


    def parse_story(self, story_text):
        # Regular expressions for the Exposition, Climax, and Resolution parts
        exposition_pattern = r"Exposition:\s*(.*?)\s*(?=Climax:|$)"
        climax_pattern = r"Climax:\s*(.*?)\s*(?=Resolution:|$)"
        resolution_pattern = r"Resolution:\s*(.*?)(?=$)"

        # Use the 're' module to find the parts in the text
        exposition = re.search(exposition_pattern, story_text, re.DOTALL)
        climax = re.search(climax_pattern, story_text, re.DOTALL)
        resolution = re.search(resolution_pattern, story_text, re.DOTALL)

        # If a part was found, get the matched text, otherwise use an empty string
        exposition_text = exposition.group(1).strip() if exposition else ""
        climax_text = climax.group(1).strip() if climax else ""
        resolution_text = resolution.group(1).strip() if resolution else ""

        # Return the story parts as a list
        return [exposition_text, climax_text, resolution_text]