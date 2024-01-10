'''
 * Copyright 2023 QuickAns
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

#!/usr/bin/env python
# coding: utf-8

import asyncio
import configparser
from graph import Graph
import os
import openai
from keys import *
from bs4 import BeautifulSoup
import time

openai.api_key = OPENAI_API_KEY
MODEL = "gpt-3.5-turbo"
PREFACE = f"I am a teaching assistant of the course Information Retrieval. A student in my course has posed the following question. Please answer the question in a empathetic manner and telling the chain of thought or related concepts if the question is advanced. Question: "
EMAIL_ID = "varun15@illinois.edu"
QUESTION_ID_TO_EMAIL_COMPLETED_FILE = "question_id_to_emails_completed.txt"

def read_previous_questions_answered():
    QuestionsToEmailsCompleted = set()
    if os.path.exists(QUESTION_ID_TO_EMAIL_COMPLETED_FILE):
        with open(QUESTION_ID_TO_EMAIL_COMPLETED_FILE, "r") as f:
            lines = f.readlines()
            for line in lines: 
                question_id, email_id = line.split(",")
                QuestionsToEmailsCompleted.add((question_id, email_id))
    return QuestionsToEmailsCompleted

async def greet_user(graph: Graph):
    user = await graph.get_user()
    if user is not None:
        print('Hello,', user.display_name)
        print('Email:', user.mail or user.user_principal_name, '\n')

async def retrieve_emails(graph: Graph):
    messagePage = await graph.get_inbox()
    if messagePage is not None and messagePage.value is not None:
        return messagePage.value

async def send_mail(graph: Graph, subject, content, email=None):

    if email is None:
        if  user is not None:
            user = await graph.get_user()
            email = user.mail or user.user_principal_name
    print(subject, content, email)
    await graph.send_mail(subject, content, email)
    print('Mail sent.\n')


def api_call(preface, question, model):
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": preface + question}
        ]
    )
    return completion.choices[0].message.content

def extract_relevant_and_uncompleted_questions(emailObjects, Questions_to_emails_completed):
    filteredQuestionIdToContent = {}
    for msg in emailObjects:
        bodyContent = msg._body._content
        parsedHtml = BeautifulSoup(bodyContent, "html.parser")
        
        heading = parsedHtml.find("h1").text
        print(heading)
        if "asked a question in Advanced Information Retrieval" in heading:
            if (msg._id, EMAIL_ID) not in  Questions_to_emails_completed:
                question = extract_question(parsedHtml)
                
                filteredQuestionIdToContent[msg._id] = [heading, question]
    return filteredQuestionIdToContent

def extract_question(parsedHtml):
    start = parsedHtml.find("p", attrs={'class':'markdown_tester'})
    question = f"{start.text}\n"
    for tr in start.find_next_siblings("p"):
        question += tr.text
        images = tr.find_all("img")
        if images:
            question += images[0].attrs['src'] + "\n"
    return question

def formulate_email_content(questionContent, answer):
    subject = f"QuickAns Assitant replies to '{questionContent[0]}'"
    emailContent = f"Hi there! I am QuickAns assistant and I am here to help you with questions on CampusWire.\n\n"
    emailContent += f"The following question was posed on CampusWire: \n\n``{questionContent[1]}``\n\n Here is a baseline response / some helpful tips to answer the question: \n\n ``{answer}`` \n\n" 
    emailContent += "Thanks, Hope you have a great day on Campus (Wire)!\n QuickAns Assistant"
    return subject, emailContent

async def main():
    print('Welcome to QuickAns!\n')
    config = configparser.ConfigParser()
    config.read(['config.cfg', 'config.dev.cfg'])
    
    azure_settings = config['azure']

    graph: Graph = Graph(azure_settings)
    await greet_user(graph)
    while True:
        QuestionsToEmailsCompleted = read_previous_questions_answered()
        emailObjects = await retrieve_emails(graph)
        filteredQuestionIdToContent = extract_relevant_and_uncompleted_questions(emailObjects, QuestionsToEmailsCompleted)

        for questionId, questionContent in filteredQuestionIdToContent.items():
            answer = api_call(PREFACE, questionContent[1], MODEL)
            subject, emailContent = formulate_email_content(questionContent, answer)
            await send_mail(graph, subject, emailContent, EMAIL_ID)
            with open(QUESTION_ID_TO_EMAIL_COMPLETED_FILE, "a") as f:
                f.write(f"{questionId},{EMAIL_ID}\n")
            # break only for demo
            # break
            time.sleep(20)
        time.sleep(120)
# Run main
asyncio.run(main())