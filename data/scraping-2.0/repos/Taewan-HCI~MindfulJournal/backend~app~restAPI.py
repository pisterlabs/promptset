# -*- coding: utf-8 -*-
from fastapi import Request, FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel  # This is the missing import

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import openai
from dotenv import load_dotenv
import os
from typing import List


SECRET_KEY = "your_secret_key"  # Make sure to use a secure and unpredictable key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

app = FastAPI()
origins = ["*", "http://localhost:3000", "http://localhost:8000", "https://mindful-journal-frontend-s8zk.vercel.app/"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 구글 Firebase인증 및 openAI api key 설정
load_dotenv()
gptapi = os.getenv("gptkey")
cred = credentials.Certificate(
    '/Users/taewankim/PycharmProjects/LLM_diary/backend/mindfuljournal-44166-firebase-adminsdk-frpg8-10b50844cf.json')
app_1 = firebase_admin.initialize_app(cred)
db = firestore.client()
My_OpenAI_key = gptapi
openai.api_key = My_OpenAI_key

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")




class TokenData(BaseModel):
    username: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(fake_db, username: str, password: str):
    user = fake_db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

class User(BaseModel):
    username: str
    hashed_password: str


class UserBase(BaseModel):
    username: str

class UserIn(UserBase):
    hashed_password: str

class User(UserBase):
    hashed_password: Optional[str] = None


def get_user(fake_db, username: str):
    if username in fake_db:
        user_dict = fake_db[username]
        return User(**user_dict)


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = fake_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return User(username=token_data.username)


from datetime import datetime


def download_diary(patient_id: str):
    doc_refs = db.collection(u'session').document(patient_id).collection(u'diary').stream()

    keys_to_keep = ['sessionStart', 'sessionEnd', 'like', 'operator', 'diary',
                    'sessionNumber', 'conversation']  # the keys you want to keep

    diaries = []
    for doc in doc_refs:
        if doc.get('isFinished') == True:  # only add diary to list if 'isFinished' is True
            diary = doc.to_dict()
            diary = {key: diary[key] for key in keys_to_keep if key in diary}

            # calculate duration if both 'sessionStart' and 'sessionEnd' are in the diary
            if 'sessionStart' in diary and 'sessionEnd' in diary:
                # assuming that 'sessionStart' and 'sessionEnd' are UNIX timestamps
                duration = diary['sessionEnd'] - diary['sessionStart']
                diary['duration'] = duration  # add 'duration' to the diary
            if 'conversation' in diary:
                count = 0
                for i in range(len(diary['conversation'])):
                    if diary['conversation'][i]['role'] == "user":
                        temp = len(diary['conversation'][i]['content'])
                        count = count + temp
                diary['length'] = count
                del diary['conversation']  # delete 'conversation' from the diary

            diaries.append(diary)

    return diaries if diaries else 'No such document!'


def download_diary_betweendate(patientID: str, startDate: int, endDate: int):
    doc_refs = db.collection(u'session').document(patientID).collection(u'diary').stream()
    keys_to_keep = ['sessionStart', 'sessionEnd', 'like', 'operator', 'diary',
                    'sessionNumber', 'conversation']  # the keys you want to keep
    diaries = []
    for doc in doc_refs:
        diary = doc.to_dict()
        if diary.get('isFinished') == True and startDate <= diary.get('sessionEnd', 0) <= endDate:  # only add diary to list if 'isFinished' is True and 'sessionEnd' is within the range
            diary = {key: diary[key] for key in keys_to_keep if key in diary}
            if 'sessionStart' in diary and 'sessionEnd' in diary:
                # assuming that 'sessionStart' and 'sessionEnd' are UNIX timestamps
                duration = diary['sessionEnd'] - diary['sessionStart']
                diary['duration'] = duration  # add 'duration' to the diary
            if 'conversation' in diary:
                count = 0
                for i in range(len(diary['conversation'])):
                    if diary['conversation'][i]['role'] == "user":
                        temp = len(diary['conversation'][i]['content'])
                        count = count + temp
                diary['length'] = count
                del diary['conversation']  # delete 'conversation' from the diary

            diaries.append(diary)

    return diaries if diaries else 'No such document!'


def frequency_betweendate(patientID: str, startDate: int, endDate: int):
    doc_refs = db.collection(u'session').document(patientID).collection(u'diary').stream()
    keys_to_keep = ['sessionEnd']  # the keys you want to keep
    diaries = []
    num = []
    for doc in doc_refs:
        diary = doc.to_dict()
        if diary.get('isFinished') == True and startDate <= diary.get('sessionEnd', 0) <= endDate:  # only add diary to list if 'isFinished' is True and 'sessionEnd' is within the range
            diary = {key: diary[key] for key in keys_to_keep if key in diary}
            diaries.append(diary)
    for i in range (len(diaries)):
        num.append(diaries[i]['sessionEnd'])
    return num if diaries else 'No such document!'




def length_betweendate(patientID: str, startDate: int, endDate: int):
    doc_refs = db.collection(u'session').document(patientID).collection(u'diary').stream()
    keys_to_keep = ['sessionStart', 'sessionEnd', 'like', 'operator', 'diary',
                    'sessionNumber', 'conversation']  # the keys you want to keep
    diaries = []
    for doc in doc_refs:
        diary = doc.to_dict()
        if diary.get('isFinished') == True and startDate <= diary.get('sessionEnd',
                                                                      0) <= endDate:  # only add diary to list if 'isFinished' is True and 'sessionEnd' is within the range
            diary = {key: diary[key] for key in keys_to_keep if key in diary}
            if 'sessionStart' in diary and 'sessionEnd' in diary:
                # assuming that 'sessionStart' and 'sessionEnd' are UNIX timestamps
                duration = diary['sessionEnd'] - diary['sessionStart']
                diary['duration'] = duration  # add 'duration' to the diary
            if 'conversation' in diary:
                count = 0
                for i in range(len(diary['conversation'])):
                    if diary['conversation'][i]['role'] == "user":
                        temp = len(diary['conversation'][i]['content'])
                        count = count + temp
                diary['length'] = count
                del diary['conversation']
                del diary['sessionStart']
                del diary['like']
                del diary['diary']
                del diary['sessionNumber']
                # delete 'conversation' from the diary

            diaries.append(diary)

    return diaries if diaries else 'No such document!'



def download_specific_diary(patient_id: str, diaryID: str):
    doc = db.collection(u'session').document(patient_id).collection(u'diary').document(diaryID).get()

    keys_to_keep = ['sessionStart', 'sessionEnd', 'operator', 'diary', 'conversation', 'phq_item_score', 'phq9score']  # the keys you want to keep

    if doc.exists:
        diary = doc.to_dict()
        if diary.get('isFinished') == True:  # only add diary to list if 'isFinished' is True
            diary = {key: diary[key] for key in keys_to_keep if key in diary}
            if 'sessionStart' in diary and 'sessionEnd' in diary:
                # assuming that 'sessionStart' and 'sessionEnd' are UNIX timestamps
                duration = diary['sessionEnd'] - diary['sessionStart']
                diary['duration'] = duration  # add 'duration' to the diary
            if 'conversation' in diary:
                count = 0
                for i in range (len(diary['conversation'])):
                    if diary['conversation'][i]['role'] == "user":
                        temp = len(diary['conversation'][i]['content'])
                        count = count + temp
                diary['length'] = count
            if 'phq_item_score' in diary:
                phqitem = []
                for i in range (len(diary["phq_item_score"])):
                    temp = diary["phq_item_score"][i]['current']
                    phqitem.append(temp)
                diary['phq_item'] = phqitem
                diary.pop("phq_item_score")

            return diary
        else:
            return 'Document "isFinished" field is False!'
    else:
        return 'No such document!'



def get_patient(patient_id: str):
    doc = db.collection(u'patient').document(patient_id).get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None

def download_all():
    docs = db.collection(u'patient').stream()

    all_data = []
    for doc in docs:
        all_data.append(doc.to_dict())

    return all_data


fake_db = {
    "expertReview": {
        "username": "expertReview",
        "hashed_password": get_password_hash("MindfulDiary0529")
    }
}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/token_test")
async def test_endpoint(current_user: User = Depends(get_current_user)):
    return {"message": "authorized"}

@app.get("/patient_all")
async def read_root(current_user: User = Depends(get_current_user)) -> dict:
    response = download_all()
    return response

@app.get("/patient/{patient_id}")
async def read_patient(patient_id: str, current_user: User = Depends(get_current_user)) -> dict:
    patient_data = get_patient(patient_id)
    if patient_data is not None:
        return patient_data
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/diary/{patient_id}")
async def read_patient(patient_id: str, current_user: User = Depends(get_current_user)) -> dict:
    diary_data = download_diary(patient_id)
    if diary_data is not None:
        return diary_data
    else:
        raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/{patientID}")
async def read_patient(patientID: str, start: int, end: int, current_user: User = Depends(get_current_user)) -> dict:
    diary_data2 = download_diary_betweendate(patientID, start, end)
    if diary_data2 != 'No such document!':
        return diary_data2
    else:
        raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/frequency/{patientID}")
async def read_patient(patientID: str, start: int, end: int, current_user: User = Depends(get_current_user)) -> dict:
    diary_data2 = frequency_betweendate(patientID, start, end)
    if diary_data2 != 'No such document!':
        return diary_data2
    else:
        raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/length/{patientID}")
async def read_patient(patientID: str, start: int, end: int, current_user: User = Depends(get_current_user)) -> dict:
    diary_data2 = length_betweendate(patientID, start, end)
    if diary_data2 != 'No such document!':
        return diary_data2
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/{patient_id}/{diaryID}")
async def read_patient(patient_id: str, diaryID: str, current_user: User = Depends(get_current_user)) -> dict:
    diary_data = download_specific_diary(patient_id, diaryID)
    if diary_data is not None:
        return diary_data
    else:
        raise HTTPException(status_code=404, detail="Patient not found")