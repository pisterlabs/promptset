from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core import VERSION
from dependencies import get_api_key, get_db
from models import User
from openai_utils import OpenAI

router = APIRouter()


class Login(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(credentials: Login, db: Session = Depends(get_db)):
    response = {"success": False}
    if credentials:
        user = User.authenticate(
            db, username=credentials.username.lower(), password=credentials.password
        )
        if user:
            response["success"] = True
            response["token"] = user.api_key
            response["user"] = user.to_sanitized_dict()
        else:
            response["error"] = "Credentials are incorrect"
    else:
        response["error"] = "Username and Password fields are mandatory"
    return response


@router.get("/version")
def version():
    return {"version": VERSION}


class RequestMessage(BaseModel):
    message: str


@router.post("/chatgpt")
def index(
    request_message: RequestMessage,
    openai: OpenAI = Depends(OpenAI),
    user: User = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    response = {"success": False}
    if request_message:
        success = user.subtract_tokens(db, len(request_message.message) / 2.5)
        if success:
            response = process_message(openai, user, request_message.message, response)
            success = user.subtract_tokens(db, len(response["last_response"]) / 2.5)
            user.delete_duplicates(db)
            response["user"] = user.to_sanitized_dict()
        else:
            response["error"] = "User {} has not enough available tokens.".format(
                user.username
            )
    else:
        response["error"] = "Message field is mandatory"
    return response


@router.post("/reset")
def reset(user: User = Depends(get_api_key), db: Session = Depends(get_db)):
    response = {"success": user.reset_session(db)}
    return response


def process_message(openai, user, message, response):
    try:
        context = openai.completion(user, message)
        response["context"] = context
        print(context[-1]["content"])
        response["last_response"] = context[-1]["content"]
        response["success"] = True
    except Exception as ex:
        response["error"] = str(ex)

    return response


@router.get("/")
def index_method_not_allowed():
    return {"detail": "Method Now Allowed", "message": "Please, use POST method"}
