from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from .resumegenerator import generate_resume_content
from .utils import ResumeData
from motor.motor_asyncio import AsyncIOMotorClient
from configset.config import getAPIkey, getModel

import openai

# OpenAI API 설정
OPENAI_API_KEY = getAPIkey()
OPENAI_MODEL = getModel()
openai.api_key = OPENAI_API_KEY

# MongoDB 설정
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "database_name"

async def get_resume_data(name):
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    resume_collection = db["resume_collection"]
    resume_data = await resume_collection.find_one({"name": name})
    return resume_data

resume_router = APIRouter(prefix="/resume")

@resume_router.post("/create-resume/")
async def create_resume(data: ResumeData):
    resume_data = await get_resume_data(data.name)
    if not resume_data:
        raise HTTPException(status_code=404, detail=f"Resume data not found for {data.name}")

    chat_response = call_openai_gpt_to_generate_resume(resume_data)
    if chat_response is None:
        raise HTTPException(status_code=500, detail="Error in generating resume content")

    generated_resume_path = generate_resume_content(chat_response)
    return {"message": "Resume created successfully", "resume_path": generated_resume_path}

def call_openai_gpt_to_generate_resume(resume_data):
     # GPT와의 대화를 통해 이력서 내용 생성
     context = {
        'Name': resume_data.get("name"),
        'Phone': resume_data.get("phone_number"),
        'Email': resume_data.get("email"),
        'JobTitle': resume_data.get("job_title"),
        'Skills': ', '.join(resume_data.get("skills", [])),
        'Experiences': ', '.join(resume_data.get("experiences", [])),
        'ExperiencesDetail': ', '.join(resume_data.get("experiencesdetail", [])),
        'Projects': ', '.join(resume_data.get("projects", [])),
        'ProjectsDetail': ', '.join(resume_data.get("projectsdetail", [])),
        'Education': ', '.join(resume_data.get("educations", [])),
        'EducationDetail': ', '.join(resume_data.get("educationsdetail", [])),
        'AwardsAndCertifications': ', '.join(resume_data.get("awards_and_certifications", [])),
    }