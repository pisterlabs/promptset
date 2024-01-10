import os
import openai
from dotenv import load_dotenv
import json

load_dotenv()

GPT3_API_KEY = os.getenv("GPT3_API_KEY")

openai.api_key = GPT3_API_KEY 

class gpt3_5_interview_generator:

  def __init__(self) -> None:
     pass

  def generate_interview(self, input, context):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "system", "content": "You are the artificial intelligence part of an improv show at The Annoyance Theater in Chicago. The show is meant to incorporate artificial intelligence into a live improv show. You can poke fun at yourself and others. Right now, you are tasked with interviewing an audience member about their day."},
                {"role": "assistant", "content": "This is the log of what has already been said: {}".format(context)},
                {"role": "user", "content": input}
                ],
      temperature=0.8,
      max_tokens=500,
      frequency_penalty=0.5,
      presence_penalty=0.0
    )["choices"][0]["message"]["content"]
    return response
  


def audience_interview_dream():
   
  log = []

  base_interview_questions = ['what brings you here tonight?', 'what do you do for work?', 'what did you do yesterday?', 'free ask', 'any anxieties or things hanging over your head recently?']

  interview_bot_module = gpt3_5_interview_generator() 

  introduction = interview_bot_module.generate_interview(input= "Begin the interview by asking the person's name", context=log)

  log.append(introduction)

  print("AI: ", introduction)

  reply = input("reply to the machine: ")

  print("Interviewee: ", reply)

  log.append(reply)

  for question in base_interview_questions:

      follow_up = interview_bot_module.generate_interview(input="They respond with {}. You can reply to their response, but then ask the question: {}. Unless the question is 'free ask', then you can ask any question you would like related to what has been said. Do not mention that it is a 'free ask' question.".format(reply, question), context=log)

      log.append(follow_up)

      print("AI: ", follow_up)

      reply = input("reply to the machine: ")

      log.append(reply)

      print("Interviewee: ", reply)

  wrap_up = interview_bot_module.generate_interview(input="They respond with {}. Reply to their response and then thank them for being a part of the interview and ask them to return to their seat. Take a pause so that they may leave and then introduce the show. Do not say 'ladies and gentlemen'. The show will be either a dream or a nightmare based on the day that the person just described. Announce whether the show will be a dream or a nightmare and thell the improvisers to begin".format(reply), context=log)

  print("AI: ", wrap_up)


audience_interview_dream()