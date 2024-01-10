from flask import Response
from openai import OpenAI
from highlight import update_highlight
from bson.objectid import ObjectId
from database import db
import json

courses = db["Courses"]

def create_completion_with_file(client, file, user_prompt):
    # Create the completion request
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an academic assistant, specialized in addressing questions specifically related to the current course. Your responses are tailored to the course content and syllabus, and you provide sources from the course material. You focus solely on assisting with course-related queries."},
            {"role": "user", "content": file},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion

def extract(s, l, r):
  result = []
  sentence = ""
  flag = False
  for c in s:
    if c == l:
      sentence = ""
      flag = True
    elif c == r:
      flag = False
      result.append(sentence)
    elif flag:
      sentence += c
  return result

def chatRespond(user_id, course_id, question):
  try:
    course = courses.find_one({"_id":ObjectId(course_id)})["name"]
    txt = courses.find_one({"_id":ObjectId(course_id)})["syllabus"]["txt"]
    pdf = courses.find_one({"_id":ObjectId(course_id)})["syllabus"]["pdf"]
  except:
    return Response(status=404)

  prompt = """
  Using the uploaded syllabus for [%s], answer [%s]. 

  Format your response as a json in this format {"answer": string, "sources": string[]}. 

  "answer" is your answer to the question. It should be succinct and helpful.

  "sources" is a list of strings that are direct quotes from the given syllabus file that support the response. 

  Split source strings if there is a new line or if they are the contents of a list or bullet points.
  """ % (course, question)

  try:
    completion = create_completion_with_file(OpenAI(), txt, prompt)
    response = json.loads(completion.choices[0].message.content)
    answer = response["answer"]
    sources = response["sources"]
  except:
    return {"answer": "Response Failed to Complete", "valid": False}

  update_highlight(user_id, pdf, sources)

  return {"answer": answer, "valid": True}
