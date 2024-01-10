# -*- coding: utf-8 -*-
"""
Filename: chatgpt.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 25.08.2023
Last Modified: 17.10.2023

Description:
This file contains testing procedures for ChatGPT experiments
"""

import string
import sys

import asyncio
from openai_python_api import ChatGPT

# pylint: disable=import-error
from examples.creds import oai_token, oai_organization  # type: ignore
from utils.audio_recorder import AudioRecorder
from utils.transcriptors import CustomTranscriptor
from utils.tts import CustomTTS

gpt = ChatGPT(auth_token=oai_token, organization=oai_organization, model="gpt-3.5-turbo")
gpt.max_tokens = 200
gpt.stream = True

tts = CustomTTS(method="google", lang="en")

# queues
prompt_queue = asyncio.Queue()
tts_queue = asyncio.Queue()


async def ask_chat(user_input):
    """
    Ask chatbot a question

    :param user_input: (str) User input

    :return: (str) Chatbot response
    """
    full_response = ""
    word = ""
    async for response in gpt.str_chat(user_input):
        for char in response:
            word += char
            if char in string.whitespace or char in string.punctuation:
                if word:
                    await prompt_queue.put(word)
                    word = ""
            sys.stdout.write(char)
            sys.stdout.flush()
            full_response += char
    print("\n")
    return full_response


async def tts_task():
    """Task to process words and chars for TTS"""
    limit = 5
    empty_counter = 0
    while True:
        if prompt_queue.empty():
            empty_counter += 1
        if empty_counter >= 3:
            limit = 5
            empty_counter = 0
        words = []
        # Get all available words
        limit_counter = 0
        while len(words) < limit:
            try:
                word = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
                words.extend(word.split())
                if len(words) >= limit:
                    break
            except asyncio.TimeoutError:
                limit_counter += 1
                if limit_counter >= 10:
                    limit = 1

        # If we have at least limit words or queue was empty 3 times, process them
        if len(words) >= limit:
            text = " ".join(words)
            await tts.process(text)
            limit = 1


async def tts_sentence_task():
    """Task to handle sentences for TTS"""
    punctuation_marks = ".?!,;:"
    sentence = ""
    while True:
        try:
            word = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
            sentence += " " + word
            # If the last character is a punctuation mark, process the sentence
            if sentence[-1] in punctuation_marks:
                await tts_queue.put(sentence)
                sentence = ""
        except Exception:  # pylint: disable=broad-except
            pass


async def tts_worker():
    """Task to process sentences for TTS"""
    while True:
        try:
            sentence = await tts_queue.get()
            if sentence:
                await tts.process(sentence)
                tts_queue.task_done()
        except Exception:  # pylint: disable=broad-except
            pass


async def get_user_input():
    """Get user input"""
    while True:
        try:
            user_input = input()
            if user_input.lower() == "[done]":
                break
            await ask_chat(user_input)
        except KeyboardInterrupt:
            break


async def main():
    """Main function"""
    asyncio.create_task(tts_sentence_task())
    asyncio.create_task(tts_worker())
    method = "google"

    while True:
        try:
            if "google" not in method:
                file_path = AudioRecorder().listen()
                with open(file_path, "rb") as f:
                    transcript = await gpt.transcript(file=f, language="en")
            else:
                transcript = CustomTranscriptor(language="en-US").transcript()
            if transcript:
                print(f"User: {transcript}")
                # translate = CustomTranslator(source='ru', target='en').translate(transcript)
                # print(translate)
                await ask_chat(transcript)
        except KeyboardInterrupt:
            break


asyncio.run(main())
