import openai
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import ast

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statistics import harmonic_mean
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('coursea_data.csv')
df.drop(['Unnamed: 0', 'course_organization'], axis=1, inplace=True)
df = df[df.course_students_enrolled.str.endswith('k')]

df['course_students_enrolled'] = df['course_students_enrolled'].apply(lambda enrolled : eval(enrolled[:-1]) * 1000)

minmax_scaler = MinMaxScaler()
scaled_ratings = minmax_scaler.fit_transform(df[['course_rating','course_students_enrolled']])

df['course_rating'] = scaled_ratings[:,0]
df['course_students_enrolled'] = scaled_ratings[:,1]
df['overall_rating'] = df[['course_rating','course_students_enrolled']].apply(lambda row : harmonic_mean(row), axis=1)

minmax_scaler = MinMaxScaler()
scaled_ratings = minmax_scaler.fit_transform(df[['course_rating','course_students_enrolled']])

df = df[df.course_title.apply(lambda title : detect(title) == 'en')]


vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df.course_title)


def recommend_by_course_title (title, recomm_count=10) : 
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:,0]))[-recomm_count:]
    sdf = df.iloc[idx].sort_values(by='overall_rating', ascending=False)
    return sdf




load_dotenv()


openai.api_key = os.environ.get("OPENAI_API_KEY")


app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def askGPT(text):
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = text,
        temperature = 0.6,
        max_tokens = 1000,
    )
    return response.choices[0].text

@app.post("/get_job/")
async def get_job(info : Request):

  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)

  Role = infoDict['Role']
  Organisation = infoDict['Organisation']
  Stipend = infoDict['Stipend']
  Qualification = infoDict['Qualification']
  Contact = infoDict['Contact']


  prompt = ( "You have been tasked with creating a job description for a new role at your organization. "
            "The following attributes have been provided: \n"
            f"1. Role: {Role} \n"
            f"2. Organization: {Organisation}  \n"
            f"3. Stipend : {Stipend} \n"
            f"4. Qualification : {Qualification} \n"
            f"5. Contact {Contact} \n\n"
            "Using Text-Davinci-003, write a job description that highlights the responsibilities and requirements of the role, as well as the benefits of working at the organization. "
            "Make sure to include the stipend, necessary qualifications, and contact information for interested applicants. "
            "Your job description should be clear, concise, and engaging, and should attract top talent to your organization.")

  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0.5,
      max_tokens=2500,
      n=1,
      stop=None,
  )

  job_description = response.choices[0].text.strip()
  
  return {"Job_Description" : job_description}



@app.post("/get_course/")
async def get_course(info : Request):
  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)

  Course_Title = infoDict['Course_Title']
  Course_Duration = infoDict['Course_Duration']
  Instructor_Name = infoDict['Instructor_Name']
  Course_Structure = infoDict['Course_Structure']

  prompt = ( "Create a comprehensive course content that covers the following attributes:"
           "The following attributes have been provided: \n"
           f"1. Course_Title: {Course_Title} \n"
           f"2. Course Duration: {Course_Duration}  \n"
           f"3. Instructor_Name: {Instructor_Name} \n"
           f"4. Course_Structure  : {Course_Structure} \n\n"
           "Use the Davinci003 model to generate a complete and coherent course content that includes all of the above attributes. The content should be well-structured and easy to follow, catering to both beginners and experts in the field. Ensure that the content is informative, engaging, and covers all the necessary topics to give students a comprehensive understanding of the subject matter.")
  
  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0.5,
      max_tokens=3000,
      n=1,
      stop=None,
  )

  Course_description = response.choices[0].text.strip()

  # TagPrompts = (
  #     "Can you generate only technical tags for content give below at max 5 tags in an array format"
  #     f"Course Content : {Course_description}"
  #     "Want the response in the below format"
  #     "Array = [List of all the tags]"
  # )

  # response = openai.Completion.create(
  #     engine="text-davinci-003",
  #     prompt=TagPrompts,
  #     temperature=0.5,
  #     max_tokens=100,
  #     n=1,
  #     stop=None,
  # )

  # Course_Tags = response.choices[0].text.strip()
  # TagList = ast.literal_eval(Course_Tags.split('=')[1][1:])

  return {"Course_description" : Course_description}  



@app.post("/get_Mentor/")
async def get_Mentor(info : Request):
  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)

  Mentor_Name = infoDict['Mentor_Name']
  Qualification = infoDict['Qualification']
  Organisation = infoDict['Organisation']
  Expertise = infoDict['Expertise']
  Contact = infoDict['Contact']

  prompt = (
      "Generate a description for a mentor using the following attributes:"
      f"Mentor Name : {Mentor_Name}"
      f"Qualification : {Qualification}"
      f"Organisation : {Organisation}"
      f"Expertise : {Expertise}"
      f"Contact : {Contact}"
      "Please use these attributes to create a detailed and informative description for the mentor. The description should highlight the mentor's achievements and experience, as well as their commitment to helping others. Please make sure that the language is professional and engaging, catering to both beginners and experts in the field"
  )


  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0.5,
      max_tokens=1024,
      n=1,
      stop=None,
  )

  Mentor_description = response.choices[0].text.strip()

  return {"Mentor_description" : Mentor_description }

@app.post("/get_chat/")
async def get_chat(info : Request):
  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)

  print(infoDict['Question'])

  return askGPT(infoDict['Question'])




@app.post("/get_course_recommendation/")
async def get_course_recommendation(info : Request):
  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)

  dk = recommend_by_course_title(infoDict['SearchString'])
  List_Courses = dk.to_dict('records')

  return {"List_Course" : List_Courses}



@app.post("/get_job_recommendation/")
async def get_job_recommendation(info : Request):
  print(await info.body())
  infoDict = await info.json()
  infoDict = dict(infoDict)






@app.get("/getInformation")
def getInformation(info : Request):
  return FileResponse("obama.mp4")
