from typing import Literal

from fastapi import APIRouter

from markji_wordcard_assistant.markji_api import api
from markji_wordcard_assistant.voice import openai_tts, elevenlab, edge
from .result import R

router = APIRouter()


@router.get("/edgetts")
async def edgetts(
        markji_token: str,
        text: str,
        voice: Literal["random", "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural",
        "en-GB-SoniaNeural", "en-GB-ThomasNeural",
        "en-US-AriaNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
        "en-US-GuyNeural", "en-US-JennyNeural", "en-US-RogerNeural", "en-US-SteffanNeural"] = 'random',
        rate: float = 1.0
):
    ret = api.upload_voice(await edge.tts(text=text,
                                          voice=voice,
                                          rate=rate),
                           markji_token=markji_token)
    return R.success(ret)


@router.get("/openai")
async def openai(
        markji_token: str,
        openai_token: str,
        text: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        = "fable",
        model: Literal["tts-1", "tts-1-hd", "tts-1-1106", "tts-1-hd-1106"]
        = "tts-1-hd-1106",
        rate: float = 1.0,
) -> R:
    ret = api.upload_voice(await openai_tts.tts(text=text,
                                                openai_token=openai_token,
                                                voice=voice,
                                                model=model,
                                                rate=rate),
                           markji_token=markji_token)
    return R.success(ret)


@router.get("/elevenlab")
async def eleven_lab(
        markji_token: str,
        eleven_token: str,
        text: str,
        voice: str = 'Adam',
        model: Literal["eleven_multilingual_v2", "eleven_monolingual_v1"] = "eleven_multilingual_v2",
) -> R:
    ret = api.upload_voice(await elevenlab.tts(text=text,
                                               elevenlab_token=eleven_token,
                                               voice=voice,
                                               model=model, ),
                           markji_token=markji_token)
    return R.success(ret)
