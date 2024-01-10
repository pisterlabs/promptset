from fastapi import Depends, APIRouter, Request
from .auth import authenticator
from pydantic import BaseModel
from models.messages import (
    UserMessage,
)
from queries.messages import ConversationQueries
from models.accounts import AccountOut
import os
import openai

# from datetime import datetime

openai.api_key = os.environ["OPENAI_API_KEY"]

router = APIRouter()


class TestResponse(BaseModel):
    message: str


class ReceivedMessage(BaseModel):
    text: str


# POST
@router.post("/api/messages", response_model=ReceivedMessage)
async def user_message_in(
    message: UserMessage,
    request: Request,
    repo: ConversationQueries = Depends(),
    account: AccountOut = Depends(authenticator.try_get_current_account_data),
):
    print(account)
    # message["token"] = "token"
    # message["role"] = "user"
    # message["timestamp"] = datetime.now()
    # repo.create(userMessageIn=message)
    # message_history = repo.get_all_messages()
    # # ensure output of message_history is a list of dictionaries
    # # CHAP GPT CHATBOT EXAMPLE REQUEST
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # messages must be a list dictionaries in the form
        # {"role": "user/system", {"content": "string"}}
        messages=[{"role": "user", "content": "Hello there!"}],
    )

    # query to write completion to the DB collection
    response = completion.choices[0].message.content

    # return chat message
    return {"text": response}


"""
import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
"""
