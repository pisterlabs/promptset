from fastapi import status, Body
import ipdb
import openai
from dotenv import load_dotenv
import os
import json

from fastapi.responses import JSONResponse
from master_ozz.ozz_query import ozz_query
from master_ozz.utils import ozz_master_root

from fastapi import APIRouter
router = APIRouter(
    prefix="/api/data",
    tags=["auth"]
)
# from fastapi import FastAPI
# router = FastAPI()

main_root = ozz_master_root()  # os.getcwd()
load_dotenv(os.path.join(main_root, ".env"))

# setting up FastAPI

# Loading the environment variables

@router.get("/test", status_code=status.HTTP_200_OK)
def load_ozz_voice():
    json_data = {'msg': 'test'}
    return JSONResponse(content=json_data)

@router.post("/voiceGPT", status_code=status.HTTP_200_OK)
def load_ozz_voice(api_key=Body(...), text=Body(...), self_image=Body(...), refresh_ask=Body(...), face_data=Body(...)): #, client_user=Body(...)):

    print(face_data)
    if api_key != os.environ.get("ozz_key"): # fastapi_pollenq_key
        print("Auth Failed", api_key)
        # Log the trader WORKERBEE
        return "NOTAUTH"
    
    client_user = 'stefanstapinski@gmail.com'

    json_data = ozz_query(text, self_image, refresh_ask, client_user)
    return JSONResponse(content=json_data)