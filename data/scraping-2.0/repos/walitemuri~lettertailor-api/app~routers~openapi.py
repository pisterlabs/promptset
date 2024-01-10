from fastapi import APIRouter, Depends, HTTPException, Header, status, FastAPI
import openai
from app.schemas import GPTRequest, GPTResponse, Content, CoverLetterForm
from sqlalchemy.orm import Session
from datetime import datetime, time
from .. import oauth2, models
from .. import database
from dotenv import load_dotenv
import os

load_dotenv(".env")
openai.api_key = os.getenv('OPENAI_API_KEY')
router = APIRouter(tags=["OpenAI API Requests"])

def update_user_access(user_id: int, db: Session):
    user = db.query(models.UserAccess).filter(models.UserAccess.user_id == user_id).first()
    if not user:
        user = models.UserAccess(user_id=user_id, count=1, last_accessed=datetime.utcnow())
        db.add(user)
        db.commit()
    else:
        if user.last_accessed.date()  < datetime.utcnow().date():
            user.count = 0
        if user.count < 15:
            user.count += 1
            user.last_accessed = datetime.utcnow()
            db.commit()
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usage limit reached.")
    
@router.post("/generate", status_code=status.HTTP_201_CREATED, response_model=GPTResponse)
async def generate_cvl(data: CoverLetterForm, gpt_request: GPTRequest, current_user: int = Depends(oauth2.get_current_user), db: Session = Depends(database.get_db)):
    resume = db.query(models.Resume).filter(models.Resume.id == current_user.resume_id).first()
    
    user_prompt = f"""You are an Expert Cover Letter Assistant you are Generating cover letters tailored to maximise the efficacy of the letter in convincing hiring managers 
depending on the job and qualifications provided. You will use language that is effective and preferred in a professional environment. 
Consider context clues from the resume/qualifications on how expereienced the applicant is (intern, expereinced) etc. In Order To optimize the output. 
Make sure the output is concise enough that it can fit on a single letter sized paper with Font size 12 in Times New Roman.
Do not resuse wording from the resume, make orginal and professional statements.
Make sure the output considers what qualifications the candidate actually has and does not claim qualifications that are in the job description yet are not evident in the resume/qualifications.
4 Paragraphs is the limit.

Qualifications/Resume:
{resume.resume_string}
Job Description:
{data.job_description}
Sending Address:
{data.sending_address}
Recieving Address:
{data.recieve_address}
"""
        
    update_user_access(current_user.id, db)
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt= user_prompt,
            max_tokens=gpt_request.max_tokens,
            n=gpt_request.n,
            stop=gpt_request.stop,
            temperature=gpt_request.temperature,
        )
        if response:
            text = response.choices[0].text
            return GPTResponse(response=Content(text=text))
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error making request to GPT API.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

