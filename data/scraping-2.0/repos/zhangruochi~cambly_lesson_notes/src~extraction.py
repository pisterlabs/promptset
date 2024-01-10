#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/cambly/src/extraction.py
# Project: /home/richard/projects/cambly/src
# Created Date: Tuesday, October 17th 2023, 2:17:52 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Thu Oct 19 2023
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2023 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2023 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from .utils import create_chunks, is_json

from tqdm import tqdm
import json
from typing import Dict, List
from collections import defaultdict

# load .env
load_dotenv()



class LessonsNoteGenerator():
    """
    A class that generates lesson notes for English tutors based on student conversations.

    Attributes:
    -----------
    transcription : str
        The transcription of the student conversation.

    Methods:
    --------
    generate_dialogue(raw_transcrib_text: str) -> Dict:
        Generates a dialogue between the student and the tutor.

    generate_knowledge_points(dialogues: str) -> Dict:
        Generates knowledge points for the tutor based on the student conversation.

    generate_comments(dialogues: str) -> str:
        Generates comments for the tutor to provide feedback to the student.
    """

    def __init__(self, transcription: str, logger):

        self.llm = ChatOpenAI(model_name=os.getenv("model"),
                              temperature=0,
                              openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.transcription = transcription
        self.logger = logger

    def json_fixer(self, text) -> str:
        """
        Fix the json format of the text using LLM
        """

        template = """The text delimited by triple single quotes is json format but contains some errors. Please fix the errors and make sure the json format is correct.
        ```{text}```

        Output: 
        """

        prompt_template = PromptTemplate.from_template(template=template)

        # initialize LLMChain by passing LLM and prompt template
        llm_chain = LLMChain(llm=self.llm,
                             prompt=prompt_template,
                             verbose=True)

        text = llm_chain.run(text)

        return text

    def generate_dialogue(self, raw_transcrib_text) -> Dict:
        """
            Generates a dialogue between a tutor and a student based on a given text.

            Args:
            - raw_transcrib_text: A string representing the raw text to generate a dialogue from.

            Returns:
            - A dictionary containing the generated dialogues in the following format:
                {
                    dialogues: [
                        {tutor: <tutor's expression 1>, student: <student's expression 1>},
                        {tutor: <tutor's expression 2>, student: <student's expression 2>},
                        {tutor: <tutor's expression 3>, student: <student's expression 3>},
                        ... 
                    ]
                }
        """

        template = """
            The text delimited by triple single quotes is a real conversation between two people, one being an English tutor and the other a student. The student want the tutor to help him improve his spoken english. Please organize the conversation between the two and use the output format I provide.

            '''{sample_text}'''
            
            IMPORTANT: Do not modify any words or phrases in the conversation, do not rephrase the sentences. Folloing the JSON format below strictly is the key to success.

            OUTPUT JSON FORMAT:
            {{
                dialogues: [
                    {{tutor: <tutor's expression 1>, student: <student's expression 1>}},
                    {{tutor: <tutor's expression 2>, student: <student's expression 2>}},
                    {{tutor: <tutor's expression 3>, student: <student's expression 3>}},
                    ... 
            }}
            Output:
            """
        prompt_template = PromptTemplate.from_template(template=template)

        # initialize LLMChain by passing LLM and prompt template
        llm_chain = LLMChain(llm=self.llm,
                             prompt=prompt_template,
                             verbose=True)

        text_chunks = create_chunks(raw_transcrib_text)

        dialogues = []
        for text in tqdm(text_chunks,
                         total=len(text_chunks),
                         desc="Generating dialogues"):
            dialogues.append(llm_chain.run(text.page_content))

        dialogues_dict = defaultdict(list)
        for _ in dialogues:

            if is_json(_):
                _ = json.loads(_)
                dialogues_dict["dialogues"].extend(_["dialogues"])
            else:
                self.logger.std_print(_)
                _ = self.json_fixer(_)

                if is_json(_):
                    self.std_print("json is fixed")
                    _ = json.loads(_)
                    dialogues_dict["dialogues"].extend(_["dialogues"])

        return dialogues_dict

    def generate_knowledge_points(self, dialogues: str) -> Dict:
        """
            Generates knowledge points from a set of dialogues between an English tutor and a student.

            Args:
            - dialogues (str): The set of dialogues between an English tutor and a student.

            Returns:
            - A dictionary containing the advanced words, phrases, and expressions that appeared in the conversation.
        """

        # Map
        map_template = """The following text delimited by triple single quotes is a set of dialogues between an english tutor and a student:

        ```
        {docs}
        ```
        
        Please identify the advanced words and phrases that appeared in the conversation. Modify each student's response to make it more authentic and more native. Maintain the integrity of the student's speech and do not break a paragraph into multiple sentences.
        IMPORTANT: Folloing the JSON format below strictly is the key to success:

        ```
        {{
            "words": [
                {{"word": <word1>, "definition": <definition1>}},
                {{"word": <word2>, "definition": <definition2>}},
                {{"word": <word3>, "definition": <definition3>}},
                ...
            ]
            "phrases": [
                {{"phrase": <phrase1>, "definition": <definition1>}},
                {{"phrase": <phrase2>, "definition": <definition2>}},
                {{"phrase": <phrase3>, "definition": <definition3>}},
                ...
            ]
            "expressions": [
                {{"original": <original response 1>, "authentic": <more authentic response 1>}},
                {{"original": <original response 2>, "authentic": <more authentic response 2>}},
                {{"original": <original response 3>, "authentic": <more authentic response 3>}},
                ...
            ]
        }}
        ```     
        Output:
        """

        map_prompt = PromptTemplate.from_template(map_template)
        llm_chain = LLMChain(llm=self.llm, prompt=map_prompt, verbose=True)
        text_chunks = create_chunks(dialogues)

        json_notes_list = []
        for text in tqdm(text_chunks,
                         total=len(text_chunks),
                         desc="Generating knowledge points"):
            json_notes_list.append(llm_chain.run(text.page_content))

        lesson_note = defaultdict(list)
        for _ in json_notes_list:

            if is_json(_):
                _ = json.loads(_)
                for key in _:
                    lesson_note[key].extend(_[key])
            else:
                self.logger.std_print(_)
                _ = self.json_fixer(_)
                if is_json(_):
                    self.std_print("json is fixed")
                    _ = json.loads(_)
                    for key in _:
                        lesson_note[key].extend(_[key])

        return lesson_note

    def generate_comments(self, dialogues: str) -> str:

        map_template = """I will give you some dialogues between students and an English teacher. Your task is to find the less authentic parts of the students' expressions and provide some comments.
        {text}

        Output:
        """
        map_prompt = PromptTemplate.from_template(map_template)

        combine_prompt = """The following text contains some less authentic expressions and detailed comments given by the English teacher. Please use this content to create a comprehensive feedback report and provide suggestions on how students can improve their English speaking skills. It is best to clearly point out the areas where the student's speech is not native and provide suggestions for improvement.
        {text}

        Output:
        """

        combine_prompt = PromptTemplate.from_template(combine_prompt)

        llm_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            combine_document_variable_name="text",
            map_reduce_document_variable_name="text",
            verbose = True)

        text_chunks = create_chunks(dialogues)
        comments = llm_chain.run(text_chunks)

        return comments

    def generate_notes(self, transcription: str) -> str:

        dialogues = self.generate_dialogue(transcription)
        dialogues_text = ""
        for _ in dialogues["dialogues"]:
            dialogues_text += "**Tutor**: " + _["tutor"] + "\n\n"
            dialogues_text += "**Student**: " + _["student"] + "\n\n"

        summary = self.generate_knowledge_points(dialogues_text)
        comments = self.generate_comments(dialogues_text)

        cache = set()

        summary_text = ""
        summary_text += "###Words:\n"
        for _ in summary["words"]:
            if _["word"] in cache:
                continue

            summary_text += "**{}**: ".format(
                _["word"]) + _["definition"] + "\n\n"

            cache.add(_["word"])

        summary_text += "###Phrases:\n"
        for _ in summary["phrases"]:
            if _["phrase"] in cache:
                continue

            summary_text += "**{}**: ".format(
                _["phrase"]) + _["definition"] + "\n\n"

            cache.add(_["phrase"])

        expression_text = ""
        for _ in summary["expressions"]:
            expression_text += "**Original**: " + _["original"] + "\n\n"
            expression_text += "**Authentic**: " + _["authentic"] + "\n\n\n"

        # combine the dialogues and summary into a single document with markdown formatting

        notes = f"""# Lesson Notes\n\n##Dialogues:\n{dialogues_text}\n\n##Advanced words and phrases:\n{summary_text}\n\n##Expressions:\n{expression_text}\n\n## Comments\n{comments}\n"""

        return notes

    def run(self, output):
        """
        Writes the notes generated by `generate_notes` to a file at the given `output` path.

        Args:
            output (str): The path to the file where the notes will be written.
        """
        with open(output, "w") as f:
            f.write(self.generate_notes(self.transcription))
