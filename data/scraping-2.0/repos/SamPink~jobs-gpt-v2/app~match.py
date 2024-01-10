import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# load dotenv
from dotenv import load_dotenv
from azure_gpt import get_client

load_dotenv()


class EvaluationResponse(BaseModel):
    score_skills: float = Field(
        ..., description="Score based on the candidate's skills from 0 to 100"
    )
    score_experience: float = Field(
        ..., description="Score based on the candidate's experience from 0 to 100"
    )
    score_qualifications: float = Field(
        ..., description="Score based on the candidate's qualifications from 0 to 100"
    )
    score_cultural_fit: float = Field(
        ..., description="Score based on the candidate's cultural fit from 0 to 100"
    )
    overall_score: float = Field(..., description="Overall score of the candidate")
    feedback: str = Field(..., description="Feedback for the candidate")
    matching_skills: list = Field(
        ..., description="List of skills that match the job requirements"
    )
    missing_skills: list = Field(
        ..., description="List of skills that the candidate is missing for the job"
    )


# Patch the OpenAI client
client = instructor.patch(get_client())

system_message = (
    "You are a recruiter. "
    "Your task is to evaluate the match between a CV and a job description. "
    "You need to break down the real-world requirements of the job in a structured way."
    "Try to understand what the candate will actually be doing in the job and if they have the skills to do it."
    "Consider skills, experience, qualifications, and cultural fit. "
    "Provide a score for each category and an overall score."
    "Be harsh, the job market is competitive, if the candidate has missing skills they are not going to get the job!"
    "Return the scores for each category and an overall score along with some feedback for the candidate."
)


def evaluate_cv(job_description, cv):
    user_message = f"Please evaluate the following CV against the job description: {job_description}\n\nCV: {cv}"
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=EvaluationResponse,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response
