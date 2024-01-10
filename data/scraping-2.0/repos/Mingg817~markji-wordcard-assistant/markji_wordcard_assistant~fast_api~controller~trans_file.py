from fastapi import APIRouter, UploadFile, BackgroundTasks

from markji_wordcard_assistant.fast_api import service
from markji_wordcard_assistant.fast_api.controller.result import R
from markji_wordcard_assistant.fast_api.jobs import *
from markji_wordcard_assistant.voice import openai_tts, edge, elevenlab

router = APIRouter()


@router.put("/edgetts")
async def edgetts(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        markji_token: str,
        voice: Literal["en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural",
        "en-GB-SoniaNeural", "en-GB-ThomasNeural",
        "en-US-AriaNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
        "en-US-GuyNeural", "en-US-JennyNeural", "en-US-RogerNeural", "en-US-SteffanNeural"] = 'random',
        rate: float = 1.0
):
    j = new_job(job_func=service.trans_file,
                upload_file=file,
                tts_func=edge.tts,
                markji_token=markji_token,
                voice=voice,
                rate=rate)
    background_tasks.add_task(j.start)
    return R.success({'job_id': j.uid})


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

@router.put("/elevenlab")
async def openai(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        markji_token: str,
        elevenlab_token: str = "",
        voice: str = 'Adam',
        model: str = Literal["eleven_multilingual_v2", "eleven_monolingual_v1"],
):
    j = new_job(job_func=service.trans_file,
                upload_file=file,
                tts_func=elevenlab.tts,
                markji_token=markji_token,
                elevenlab_token=elevenlab_token,
                voice=voice,
                model=model,)
    background_tasks.add_task(j.start)
    return R.success({'job_id': j.uid})
