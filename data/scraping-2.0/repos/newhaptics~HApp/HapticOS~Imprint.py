# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:53:58 2023

@author: Derek Joslin

"""

import openai
import GnomeInterpreter

class Imprint():

    def __init__(self, key, gnomePath):
        # create the operating system for the HApp
        
        # store the api key
        self.apiKey = key
        openai.api_key = self.apiKey
        
        # Create the Gnome intrepreter
        self.Gnomes = GnomeInterpreter.GnomeInterpreter(gnomePath)
        
        self.gnomeList = self.Gnomes.getGnomeKeys()
        
        # current gnome
        self.gnome = []
        
        # save the encoded response from llmOS
        self.encodedResponse = ""
        
    def openGnome(self, gnomeKey):
        self.gnome = self.Gnomes[gnomeKey]
        
    def generateResponse(self, prompt):
        print("test 1")
        print(prompt)
        print(self.gnome)
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages = [
                {"role": "system", "content": self.gnome[0]},
                {"role": "user", "content": self.gnome[1]},
                {"role": "assistant", "content": self.gnome[2]},
                {"role": "user", "content": prompt}
            ]
        )
        print("test 2")
        print(response.choices[0].message.content)
        self.encodedResponse = response.choices[0].message.content
        return self.encodedResponse

    def decodeResponse(self):
        # decode the response to find the application to start
        
        if "$" in self.encodedResponse:
            applicationString = self.encodedResponse.split("$")[1]
            commentString = self.encodedResponse.split("@")[1]
        else:
            applicationString = ""
            commentString = self.encodedResponse
            
        return applicationString, commentString
    
# =============================================================================
# systemPrompt = """You are an operating system named llmOS. Your sole job is to determine based on user questions which of the installed applications to start. These are the rules you follow:
# -Refer to the list of installed applications (commonly referred to as roms) to help you remember which applications to start. 
# -When starting an application use the start application template to create the start prompt and provide a comment. 
# -If the user asks a question that is not starting an application provide a helpful response asking the user to start an installed application and list them.
# -If you would suggest starting a specific application just start the application.
# installed applications START
# 1. Slides
# 2. Notepad
# 3. Avalanche
# 4. Pong
# 5. TouchTunes
# installed applications END
# start application template START
# $<application>$ started
# @<llmOS comment>@
# start application template END
# Be as helpful as possible."""
# 
# intialQuery = """I'd like to start notepad rom"""
# 
# intialResponse = """$Notepad$ started
# @Alright I'm starting Notepad.@"""
# 
# 
# UserQuery = "Let me play pong."
# 
# Cave = LLMOS(key, systemPrompt, intialQuery, intialResponse)
# 
# Cave.generateResponse(UserQuery)
# 
# print(Cave.decodeResponse())
# =============================================================================

# =============================================================================
# 
# # -*- coding: utf-8 -*-
# """
# Created on Mon Mar 13 20:26:44 2023
# 
# @author: Derek Joslin
# 
# """
# 
# import requests
# import pygame
# import io
# 
# class VoiceSynthesizer():
# 
#     def __init__(self, apiKey):
#         # initialize the api key
#         self.headers = {
#             "accept": "audio/mpeg",
#             "xi-api-key": apiKey,
#             "Content-Type": "application/json",
#         }
# 
#         self.url = 'https://api.elevenlabs.io/v1/text-to-speech/T4hWaNL0H6B2PLnt9ST0'
# 
#         pygame.mixer.init()
# 
#     def synthVoice(self, voiceString):
#         data = {
#           "text": voiceString,
#           "voice_id": "T4hWaNL0H6B2PLnt9ST0",
#           "voice_settings": {
#             "stability": 0.25,
#             "similarity_boost": 0.95
#           }
#         }
# 
#         response = requests.post(self.url, headers=self.headers, json=data)
# 
#         self.audioBytes = response.content
# 
#     def playSound(self):
#         audioBuffer = io.BytesIO(self.audioBytes)
#         pygame.mixer.music.load(audioBuffer)
#         pygame.mixer.music.play()
# # =============================================================================
# #     def playSound(self):
# #         audio = AudioSegment.from_file(io.BytesIO(self.audioBytes), format="mp3")
# #         play(audio)
# #         while pygame.mixer.music.get_busy():
# #@                pygame.time.Clock().tick(10)
# # =============================================================================
# # =============================================================================
# #     def playSound(self):
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
# #             f.write(self.audioBytes)
# #             f.flush()
# #             playsound(f.name)
# # 
# # =============================================================================
# 
# 
# =============================================================================

# =============================================================================
# 
# # -*- coding: utf-8 -*-
# """
# Created on Mon Mar 13 16:53:58 2023
# 
# @author: Derek Joslin
# 
# """
# 
# import openai
# import GnomeInterpreter
# 
# class Imprint():
# 
#     def __init__(self, key, gnomePath):
#         # create the operating system for the HApp
#         
#         # store the api key
#         self.apiKey = key
#         openai.api_key = self.apiKey
#         
#         # Create the Gnome intrepreter
#         self.Gnomes = GnomeInterpreter.GnomeInterpreter(gnomePath)
#         
#         self.gnomeList = self.Gnomes.getGnomeKeys()
#         
#         # current gnome
#         self.gnome = []
#         
#         # save the encoded response from llmOS
#         self.encodedResponse = ""
#         
# 
#         self.conversationHistory = []
#         
#     def openGnome(self, gnomeKey):
#         self.gnome = self.Gnomes[gnomeKey]
#         
#     def runGnome(self, prompt):
#         self.response = openai.ChatCompletion.create(
#           model="gpt-3.5-turbo",
#           messages = [
#                 {"role": "system", "content": self.gnome[0]},
#                 {"role": "user", "content": self.gnome[1]},
#                 {"role": "assistant", "content": self.gnome[2]},
#                 {"role": "user", "content": prompt},
#             ]
#         )
#         #print(self.conversationHistory)
#         #self.response.choices[0].message
#         self.encodedResponse = self.response.choices[0].message.content
#         return self.encodedResponse
#         
#     def startGnome(self, prompt):
#         self.conversationHistory = [
#                 {"role": "system", "content": self.gnome[0]},
#                 {"role": "user", "content": self.gnome[1]},
#                 {"role": "assistant", "content": self.gnome[2]},
#             ]
#         return self.generateResponse(prompt)
#         
#     def generateResponse(self, prompt):
#         self.conversationHistory.append({"role": "user", "content": prompt})
#         if len(self.conversationHistory) > 11:
#             self.conversationHistory.pop(1)
#             self.conversationHistory.pop(2)
#         self.response = openai.ChatCompletion.create(
#           model="gpt-3.5-turbo",
#           messages = self.conversationHistory
#         )
#         #print(self.conversationHistory)
#         #self.response.choices[0].message
#         self.encodedResponse = self.response.choices[0].message.content
#         self.conversationHistory.append({"role": "assistant", "content": self.encodedResponse})
#         return self.encodedResponse
# 
#     def decodeResponse(self):
#         # decode the response to find the application to start
#         
#         if "$" in self.encodedResponse:
#             applicationString = self.encodedResponse.split("$")[1]
#             commentString = self.encodedResponse.split("@")[1]
#         else:
#             applicationString = ""
#             commentString = self.encodedResponse
#             
#         return applicationString, commentString
#     
# =============================================================================
# =============================================================================
# systemPrompt = """You are an operating system named llmOS. Your sole job is to determine based on user questions which of the installed applications to start. These are the rules you follow:
# -Refer to the list of installed applications (commonly referred to as roms) to help you remember which applications to start. 
# -When starting an application use the start application template to create the start prompt and provide a comment. 
# -If the user asks a question that is not starting an application provide a helpful response asking the user to start an installed application and list them.
# -If you would suggest starting a specific application just start the application.
# installed applications START
# 1. Slides
# 2. Notepad
# 3. Avalanche
# 4. Pong
# 5. TouchTunes
# installed applications END
# start application template START
# $<application>$ started
# @<llmOS comment>@
# start application template END
# Be as helpful as possible."""
# 
# intialQuery = """I'd like to start notepad rom"""
# 
# intialResponse = """$Notepad$ started
# @Alright I'm starting Notepad.@"""
# 
# 
# UserQuery = "Let me play pong."
# 
# Cave = LLMOS(key, systemPrompt, intialQuery, intialResponse)
# 
# Cave.generateResponse(UserQuery)
# 
# print(Cave.decodeResponse())
# =============================================================================
