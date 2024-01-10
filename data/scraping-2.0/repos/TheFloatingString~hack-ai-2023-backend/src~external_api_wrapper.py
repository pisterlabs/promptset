from src.audio_functions import generate_audio

import requests

from dotenv import load_dotenv
import os
import re
import json
from src.embeddings import generate_embeddings
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load URLs from json
with open('static/urls.json') as f:
    urls = json.load(f)

class ExternalWrapper:

    def __init__(self, name, interests, topic, modifier, narration, character_environment, character_descriptors, openai_obj):
        self.name = name
        self.interests = interests
        self.topic = topic
        self.modifier = modifier
        self.narration = narration
        self.character_environment = character_environment
        self.character_descriptors = character_descriptors
        self.openai_obj = openai_obj

        self.S1_PROMPT = ""

    def return_response(self):

        return_dict = {
            "data": {
                "s1": {
                    "title": None,
                    "text": None,
                    "highlightedText": None,
                    "imageUrl": None,
                    "audioUrl": None
                },
                "s2": {
                    "title": None,
                    "text": None,
                    "highlightedText": None,
                    "imageUrl": None,
                    "audioUrl": None
                },
                "s3": {
                    "title": None,
                    "text": None,
                    "highlightedText": None,
                    "imageUrl": None,
                    "audioUrl": None
                },
                "question": {
                    "text": None
                },
                "s4": { 
                    "images": [],
                    "text": []
                }
            }
        }


        ### S1 GENERATION ###

        s1_raw_text = self.generate_text_s1()

        print(">> Generated s1_raw_text")


        return_dict["data"]["s1"]["text"] = s1_raw_text.replace("<b>", "").replace("</b>", "")
        # Split the paragraph into sentences
        sentences = s1_raw_text.split(". ")



        ### S4 TEXT AND IMAGE GENERATION ###

        # Store each sentence in a dictionary with keys being the sentence number
        script = {str(i+1): sentence for i, sentence in enumerate(sentences)}
        s4_generated_images = generate_embeddings(script, urls)

        return_dict["data"]["s4"]["text"] = s4_generated_images[0]
        return_dict["data"]["s4"]["images"] = s4_generated_images[1]

        print(">> Generated s4 text and images")



        ### QUESTION GENERATION ###

        return_dict["data"]["s1"]["highlightedText"] = re.findall("<b>(.*?)</b>", s1_raw_text)

        return_dict["data"]["question"]["text"] = self.generate_question()

        print(">> Generated question")


        ### S2 TEXT GENERATION ###

        s2_raw_text = self.generate_text_s2(
            return_dict["data"]["question"]["text"],
            return_dict["data"]["s1"]["text"]
            )

        return_dict["data"]["s2"]["text"] = s2_raw_text.replace("<b>", "").replace("</b>", "")

        return_dict["data"]["s2"]["highlightedText"] = re.findall("<b>(.*?)</b>", s2_raw_text)

        print(">> Generated s2 text")


        ### S3 TEXT GENERATION ###

        return_dict["data"]["s3"]["text"] = self.generate_text_s3(return_dict["data"]["s1"]["text"])

        print(">> Generated s3 text")



        ### IMAGE GENERATION FOR S1, S2 AND S3 ###

        s1_s2_s3_script = {
            0: return_dict["data"]["s1"]["text"].replace("\n", ""),
            1: return_dict["data"]["s2"]["text"].replace("\n", ""),
            2: return_dict["data"]["s3"]["text"].replace("\n", "")
        }


        s1_s2_s3_generation = generate_embeddings(s1_s2_s3_script, urls)


        return_dict["data"]["s1"]["imageUrl"] = s1_s2_s3_generation[1][0]
        return_dict["data"]["s2"]["imageUrl"] = s1_s2_s3_generation[1][1]
        return_dict["data"]["s3"]["imageUrl"] = s1_s2_s3_generation[1][2]


        print(">> Generated images for s1, s2 and s3")

        ### AUDIO GENERATION ###

        generate_audio(
            text_input=return_dict["data"]["s1"]["text"].replace("\n", ""),
            filepath="static/api/audio/s1/output.mp3"
        )
        return_dict["data"]["s1"]["audioUrl"] = "/api/audio/s1"
        return_dict["data"]["s2"]["audioUrl"] = "/api/audio/s2"
        return_dict["data"]["s3"]["audioUrl"] = "/api/audio/s3"
        return_dict["data"]["question"]["audioUrl"] = "/api/audio/question"

        generate_audio(
            text_input=return_dict["data"]["question"]["text"].replace("\n", ""),
            filepath="static/api/audio/question/output.mp3"
        )

        generate_audio(
            text_input=return_dict["data"]["s2"]["text"].replace("\n", ""),
            filepath="static/api/audio/s2/output.mp3"
        )

        generate_audio(
            text_input=return_dict["data"]["s3"]["text"].replace("\n", ""),
            filepath="static/api/audio/s3/output.mp3"
        )

        print(">> Generated audio")

        ### TITLE GENERATION ###

        return_dict["data"]["s1"]["title"] = self.generate_title(return_dict["data"]["s1"]["text"].replace("\n", ""))
        return_dict["data"]["s2"]["title"] = self.generate_title(return_dict["data"]["s2"]["text"].replace("\n", ""))
        return_dict["data"]["s3"]["title"] = self.generate_title(return_dict["data"]["s3"]["text"].replace("\n", ""))


        print(">> Generated titles")


        return return_dict



    def generate_title(self, text):

        completion = self.openai_obj.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Generate a title for the following text: \"{text}\""},
            ]
        )

        return completion["choices"][0]["message"]["content"]


    def generate_text_s1(self):

        # TODO 150 words?

        self.S1_PROMPT = f"""
            Write a 150-word story story that teaches a child about the following topic: {self.topic}. 
            The story is centered around a character called {self.name} with the following characteristics: {", ".join(self.character_descriptors)}. 
            The story is set in the universe of {self.character_environment}. 
            Write the story with the following style: {self.modifier}.
            Enclose the best sentence that summarizes the topic with <b> at the start, and </b> at the end.
            The story should start with an introduction, describe the character, and include a definition of the topic. The story is around 90 words long.
            """

        resp = self.openai_obj.Completion.create(
            model="text-davinci-003",
            prompt=self.S1_PROMPT,
            max_tokens=350,
            temperature=0
            )
        
        return resp["choices"][0]["text"]


    def generate_text_s2(self, question, s1_generated_text):
        
        completion = self.openai_obj.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.S1_PROMPT},
                {"role": "system", "content": s1_generated_text},
                {"role": "user", "content": f"Write a brief paragraph using the previous story to provide a simpler explanation for the topic: {self.topic}."}
            ]
        )

        return completion["choices"][0]["message"]["content"]

    def generate_text_s3(self, s1_generated_text):
        
        completion = self.openai_obj.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.S1_PROMPT},
                {"role": "system", "content": s1_generated_text},
                {"role": "user", "content": "Generate a kind ending to this story."}
            ]
        )

        return completion["choices"][0]["message"]["content"]

    def generate_question(self):
        PROMPT = f"""
        Write a yes or no question about {self.topic}, where the correct answer should be yes. The question should be clear and easy for a child to understand.
        """

        resp = self.openai_obj.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": PROMPT}
            ],
            max_tokens=350,
            temperature=0
        )

        return resp["choices"][0]["message"]["content"]
