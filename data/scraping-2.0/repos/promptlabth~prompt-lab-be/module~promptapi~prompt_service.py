"""
Router for language model service 
- OPENAI
- VERTEX AI
"""

import os 
import random

from fastapi import APIRouter, Depends, HTTPException, Header, status, Request, Response
import openai
from pydantic import BaseModel
import dotenv
from middlewares import authentication
from fastapi.responses import JSONResponse

from typing import Annotated, List


from sqlmodel import select

from model import database
from model.promptMessages import prompt_messages_model
from model.users import users_model
from model.tones import tone_model
from model.features import features_model
from datetime import datetime

from module.promptapi.prompt_utils.repository import getFeaturById, getLanguageById, getMaxMessageByUserId, getMessagesThisMonth, getToneById, getUserByFirebaseId, getModelAIById
from module.promptapi.prompt_utils.open_ai import openAiGenerate
from module.promptapi.prompt_utils.vertex_parameter import vertexGenerator
from model.models import models_model



dotenv.load_dotenv()

class OpenAiRequest(BaseModel):
    """
    this calss is model for Request to api
    """
    input_message: str
    tone_id: int
    feature_id: int

class ResponseHttp(BaseModel):
    """
    this model for response case in endpoint
    """
    reply: str
    error: str


router = APIRouter(
    tags=["Language AI Service"],
    responses={404: {"description" : "Not Found"}}
)


class Message(BaseModel):
    id:int
    user_id:int
    tone_id:int
    tone:str
    date_time: datetime
    feature_id:str
    feature:str
    input_message:str
    result_message: str

openai.api_key = os.environ.get("OPENAI_KEY")


@router.post("/generate-random")
def generateTextReasult(
    response: Response,
    userReq: OpenAiRequest,
    firebaseId: Annotated[str, Depends(authentication.auth_depen_new)],
    Authorization:str = Header(default=None), 
    RefreshToken:str = Header(default=None),
):
    """
    In this function is will be return a old message of user by userid
    """

    model_language_choices = ["GPT", "VERTEX"]
    weights = [0.8, 0.2]
    modelLanguage = random.choices(model_language_choices, weights, k=1)[0]


    # modelLanguage = random.choices(model_language_choices, weights, k=1)[0]
    
    # if / else check a env is deploy yep?
    if os.environ.get("DEPLOY") == "DEV":
        modelLanguage = "VERTEX"
    print(firebaseId)
    user = getUserByFirebaseId(firebaseId)
    if(user == False):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseHttp(
                reply="การเข้าสู่ระบบมีปัญหา กรุณา Login ใหม่อีกครั้ง",
                error="Firebase Login is Exp"
            ).dict()
        )
    
    # handle when user limit message per day
    enableLimitMessage = True
    if(enableLimitMessage):
        total_messages_this_month = getMessagesThisMonth(user)
        mexMessage = getMaxMessageByUserId(user)
        if(total_messages_this_month >= mexMessage):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=ResponseHttp(
                    reply="คุณใช้งานเกินจำนวนที่กำหนดแล้ว กรุณาลองใหม่ในวันถัดไป",
                    error="limit message"
                ).dict()
        )
    
    
    # get tone by id
    tone = getToneById(userReq.tone_id)
    if(tone == False):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseHttp(
                reply="กรุณาลองใหม่ในภายหลัง",
                error="cannot create and save to db"
            ).dict()
        )
    
    # get language by id
    language = getLanguageById(tone.language_id)
    if(language == False):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseHttp(
                reply="กรุณาลองใหม่ในภายหลัง",
                error="cannot create and save to db"
            ).dict()
        )
    
    # get feature by id
    feature = getFeaturById(userReq.feature_id)
    if(feature == False):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseHttp(
                reply="กรุณาลองใหม่ในภายหลัง",
                error="cannot create and save to db"
            ).dict()
        )
    
    result = "กรุณาลองใหม่ในภายหลัง"

    if(modelLanguage == "GPT"):
        try:
            result = openAiGenerate(language.language_name, feature.name, tone.tone_name, userReq.input_message)
            model = getModelAIById("GPT")
        except:
            result = vertexGenerator(language.language_name, feature.name, tone.tone_name, userReq.input_message)
            model = getModelAIById("VERTEX")
    elif(modelLanguage == "VERTEX"):
        try:
            result = vertexGenerator(language.language_name, feature.name, tone.tone_name, userReq.input_message)
            model = getModelAIById("VERTEX")
        except:
            result = openAiGenerate(language.language_name, feature.name, tone.tone_name, userReq.input_message)
            model = getModelAIById("GPT")

    try:
        prompt_message_db = prompt_messages_model.Promptmessages(
            input_message=userReq.input_message,
            result_message=result,
            feature=feature,
            tone=tone,
            user=user,
            model=model,
            date_time=datetime.now()
        )
    except Exception as e:
        print("Error MSG", user)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseHttp(
                reply=result,
                error="cannot create and save to db",
            ).dict()
        )
    with database.session_engine() as session:
        try:
            session.add(prompt_message_db)
            session.commit()
            session.refresh(prompt_message_db)
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=ResponseHttp(
                    reply=result,
                    error="cannot save to db",
                    msgError=e
                ).dict()
            )
    try:
        response_data = JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ResponseHttp(reply=result, error="").dict(),
            headers={
                "AccessToken":response.headers["access-token"],
                "RefreshToken":response.headers["refresh-token"]
            }
        )
    except:
        response_data=JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ResponseHttp(reply=result, error="").dict(),
        )
        
    return response_data

@router.get("/get-caption", status_code=200)
def get_old_caption_by_user(
    # userid,
    response: Response,
    user: Annotated[str, Depends(authentication.auth_depen_new)],
    Authorization: str = Header(default=None),
    RefreshToken: str = Header(default=None),
):
    """
    In this function is will be return a old message of user by userid
    """

    messages = []
    # print(userid)


    with database.session_engine() as session:
        
        # Find id of user by firebase id
        
        try:
            statement_prompt = select(users_model.Users).where(
                users_model.Users.firebase_id == user
            ) 
            user_exec = session.exec(statement=statement_prompt).one()
            # print("userid = ", user_exec.id)
        except:
            return JSONResponse(
                content={
                        "error": "Not found a user by userid"
                },
                status_code=status.HTTP_404_NOT_FOUND
            )


        # Find a all prompt by userid
        try:
            statement_prompt = select(prompt_messages_model.Promptmessages).where(
                prompt_messages_model.Promptmessages.user_id == user_exec.id

            ) 
            prompt_messages_by_id = session.exec(statement=statement_prompt).all()
        except:
            return JSONResponse(
                content={
                        "error": "Not found a Prompt by userid"
                },
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        try:
            for prompt in prompt_messages_by_id:
                ms = Message(
                    id=prompt.id,
                    feature_id=prompt.feature_id,
                    feature=prompt.feature.name,
                    date_time=prompt.date_time,
                    input_message=prompt.input_message,
                    result_message=prompt.result_message,
                    tone_id=prompt.tone_id,
                    tone=prompt.tone.tone_name,
                    user_id=prompt.user_id
                )
                # print(ms)
                messages.append(ms)

            
            # print(messages)
            return messages
        
        except:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Found a error // you not have a prompt message, get one?"
            )
            
@router.get("/get-count-message", status_code=200)
def get_count_message(
    firebaseId: Annotated[str, Depends(authentication.auth_depen_new)],
):
    user = getUserByFirebaseId(firebaseId)
    total_messages_this_month = getMessagesThisMonth(user)
    
    return total_messages_this_month

    
@router.get("/max-message", status_code=200)
def get_max_message(
    firebaseId: Annotated[str, Depends(authentication.auth_depen_new)],
):
    user = getUserByFirebaseId(firebaseId)
    maxMessage = getMaxMessageByUserId(user)
    
    return maxMessage
    
@router.get("/remaining-message", status_code=200)
def get_remaining_message(
    firebaseId: Annotated[str, Depends(authentication.auth_depen_new)],
):
    user = getUserByFirebaseId(firebaseId)
    maxMessage = getMaxMessageByUserId(user)
    total_messages_this_month = getMessagesThisMonth(user)
    
    return maxMessage - total_messages_this_month

    
    
