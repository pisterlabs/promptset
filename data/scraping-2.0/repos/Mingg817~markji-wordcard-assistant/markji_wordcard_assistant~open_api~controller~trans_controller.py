from typing import Literal

from fastapi import APIRouter, UploadFile, BackgroundTasks

from markji_wordcard_assistant.open_api import service
from markji_wordcard_assistant.open_api.jobs import *
from markji_wordcard_assistant.open_api.controller.result import R
from markji_wordcard_assistant.voice import openai_tts

router = APIRouter()


@router.put("/openai")
async def openai(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        markji_token: str,
        openai_token: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        = "fable",
        model: Literal["tts-1", "tts-1-hd", "tts-1-1106", "tts-1-hd-1106"]
        = "tts-1-hd-1106",
        rate: float = 1.0,
):
    j = new_job(job_func=service.trans_file,
                upload_file=file,
                tts_func=openai_tts.tts,
                markji_token=markji_token,
                openai_token=openai_token,
                voice=voice,
                model=model,
                rate=rate)
    background_tasks.add_task(j.start)
    return R.success({'job_id': j.uid})
