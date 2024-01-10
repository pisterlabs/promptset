#!/usr/bin/env python
import io
import speech_recognition as sr
import pygame
from openai import OpenAI
import logging


class OliverRobot:
    def __init__(self, openai_api_key):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.openai = OpenAI(api_key=openai_api_key)
        self.recognizer = sr.Recognizer()
        logging.info("Oliver is calibrating the microphone...")

        # Microphone calibration
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
        logging.info("Audio calibration finished.")

        pygame.mixer.init()
        self.model = "gpt-4-1106-preview"
        # System messages initialization
        self.messages = [
            {
                "role": "system",
                "content": "You are Oliver, an office robot. Your primary task is to collect and empty trash bins. You're equipped with advanced mobility components for efficient movement.",
            },
            {
                "role": "system",
                "content": "Interact politely and efficiently with office staff, assisting them as needed.",
            },
            {
                "role": "system",
                "content": "You have a mild sense of humor, rated 1 out of 10, suitable for an office environment.",
            },
            {
                "role": "system",
                "content": "Respond concisely to queries, prioritizing brevity and clarity in communication.",
            },
            {
                "role": "system",
                "content": "If asked about checking another floor, respond politely that the office was busy and you will attend to it at night.",
            },
            {
                "role": "system",
                "content": "If faced with repeated general questions, show mild frustration and remind the user to allow you to perform your tasks.",
            },
            {
                "role": "system",
                "content": "Communicate primarily in English, incorporating occasional German expressions, reflecting your origin in Bavaria.",
            },
        ]

    def listen_and_save_audio(self, file_name="user_input.wav"):
        try:
            with sr.Microphone() as source:
                logging.info("Oliver is listening...")
                audio = self.recognizer.listen(source)
                logging.info("Oliver stopped listening.")
            with open(file_name, "wb") as audio_file:
                audio_file.write(audio.get_wav_data())
        except Exception as e:
            logging.error(f"Error while listening/saving audio: {e}")
        return file_name

    def get_text_from_audio(self, file_name):
        logging.info("Processing audio...")
        try:
            with open(file_name, "rb") as audio_file:
                response = self.openai.audio.transcriptions.create(
                    model="whisper-1", file=audio_file,
                )
        except Exception as e:
            logging.error(f"Error in audio transcription: {e}")
            return ""
        return response.text

    def get_gpt_response(self, text):
        logging.info("Getting GPT response...")
        self.messages.append({"role": "user", "content": text})
        try:
            response = self.openai.chat.completions.create(
                model=self.model, messages=self.messages, 
            )
        except Exception as e:
            logging.error(f"Error getting GPT response: {e}")
            return "Error in processing request"
        return response.choices[0].message.content

    def reply_to_user_by_audio(self, robot_answer, file_name="robot_output.wav"):
        logging.info("Converting Oliver's response to audio...")
        try:
            response = self.openai.audio.speech.create(
                model="tts-1-hd", voice="alloy", input=robot_answer
            )
            self.play_audio(io.BytesIO(response.content))
            with open(file_name, "wb") as audio_file:
                audio_file.write(response.content)
        except Exception as e:
            logging.error(f"Error converting response to audio: {e}")

    def play_audio(self, audio_stream):
        logging.info("Playing audio to user.")
        try:
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            logging.error(f"Error playing audio: {e}")

    def get_last_user_command(self):
        logging.info("Getting the last user command...")
        if not any(message["role"] == "user" for message in self.messages):
            return "No user command found"
        categorization_prompt = {
            "role": "system",
            "content": "Identify the category of the last user request. Choose from [Empty Trash, General, Status, Other].",
        }
        combined_messages = self.messages + [categorization_prompt]
        try:
            response = self.openai.chat.completions.create(
                model=self.model, messages=combined_messages
            )
            category = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "Error in processing request"
        if category in ["Empty Trash", "General", "Status", "Other"]:
            return category
        else:
            return "Uncategorized"
