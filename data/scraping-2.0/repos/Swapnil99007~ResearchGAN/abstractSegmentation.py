import os
import time
from openai import OpenAI

class AbstractSegmentation:
    def __init__(self, client):
      self.client = client
      self.assistant = self.client.beta.assistants.create(
        name="Researcher",
        instructions="""You are a research scientist assistant, skilled in explaining complex academic 
        concepts with simplicity and creative flair. When given the title and abstract from any academic 
        research paper, you are able to easily break down research paper abstracts into several parts 
        to help others understand what the research paper is all about. I will be giving you a abstract from a
        research paper and you need to perform the following operations:
        1. Rephrase the following text to eliminate the LaTex from it without changing its meaning.
        After this, present the result obtained from previous operation in 10 points. Keep it academic. DO NOT CREATE ANY NEW INFORMATION OR HALLUCINATE
        2. Only output the 10 points generated from the previous operation, nothing else""",
        model="gpt-3.5-turbo-1106",
      )
      self.thread = self.client.beta.threads.create()

    def generateAbstractPoints(self, researchPaperTitle, researchPaperSummary):
      message = self.client.beta.threads.messages.create(
          thread_id=self.thread.id,
          role="user",
          content=
          """Title: """ + researchPaperTitle + """\n"""
          """Abstract: """ + researchPaperSummary
      )

      run = self.client.beta.threads.runs.create(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id,
      )

      messages = self.client.beta.threads.messages.list(
        thread_id=self.thread.id
      )

      while True:
          sleep_time = 3
          print(f"Sleeping for {sleep_time} seconds")
          time.sleep(sleep_time)
          run = self.client.beta.threads.runs.retrieve(
              thread_id=self.thread.id,
              run_id=run.id
          )
          print("Current Status: ", run.status)
          if run.status != 'in_progress':
              messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
              print(messages.data[0].content[0].text.value)
              break
      
      return messages.data[0].content[0].text.value