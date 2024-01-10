'''lib import'''
from fastapi import APIRouter, Request, Depends, Form, Body, status
from starlette.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from typing import Optional
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models.database import connect_db
from models.company import Company
from promptai.langopen import LangChainAI


class TranslationRequest(BaseModel):
    '''basemodel translator'''
    prompt: str
    id: Optional[int] = None
    title: Optional[str] = None


router = APIRouter()
jin_template = Jinja2Templates(directory="templates")

@router.get("/", name="home")
def query_home(
    request: Request,
    db: Session = Depends(connect_db)
):
    '''get all AI queries'''
    companies_data = db.query(Company).all()
    result_list = []
    # Loop through the data and create dictionaries
    for item in companies_data:
        result_dict = {
            "id": item.id,
            "prompt": item.prompt,
            "title": item.title,
        }
        result_list.append(result_dict)
    context = {"request": request, "data": result_list}
    return jin_template.TemplateResponse("entry.html", context)


@router.get("/ai_query")
def query_ai_get(
    request: Request,
    db: Session = Depends(connect_db)
):
    '''get all AI queries'''
    companies_data = db.query(Company).all()
    result_list = []
    # Loop through the data and create dictionaries
    for item in companies_data:
        result_dict = {
            "id": item.id,
            "prompt": item.prompt,
            "title": item.title,
        }
        result_list.append(result_dict)
    context = {"request": request, "data": result_list}
    return jin_template.TemplateResponse("index.html", context)


@router.post("/ai_query")
def query_ai_post(
    request: Request,
    prompt: TranslationRequest,
    db: Session = Depends(connect_db),
):
    """query maker"""
    new_company = Company(prompt=prompt.prompt, title=prompt.title)
    print(prompt)
    db.add(new_company)
    db.commit()
    db.refresh(new_company)
    return {"status": "success", "data": new_company}

    # context = {"request": request, "data": new_company}
    # return jin_template.TemplateResponse("index.html", context)

@router.get("/update/{prompt_id}")
def update(
    request: Request,
    prompt_id: int,
    db: Session = Depends(connect_db)
):
    data = db.query(Company).filter(Company.id == prompt_id).first()
    context = {"request": request, "data": data}
    return jin_template.TemplateResponse("update.html", context)


@router.post("/update/{prompt_id}")
def update_prompt(
    request: Request,
    prompt: TranslationRequest,
    prompt_id: int,
    db: Session = Depends(connect_db)
):
    print(prompt)
    company = db.query(Company).filter(Company.id == prompt_id).first()
    if prompt.prompt:
        company.prompt = prompt.prompt
    if prompt.title:
        company.title = prompt.title
    db.commit()

    return {"status": "success"}


@router.get("/delete/{prompt_id}")
def delete(request: Request, prompt_id: int, db: Session = Depends(connect_db)):
    todo = db.query(Company).filter(Company.id == prompt_id).first()
    db.delete(todo)
    db.commit()
    url = router.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)