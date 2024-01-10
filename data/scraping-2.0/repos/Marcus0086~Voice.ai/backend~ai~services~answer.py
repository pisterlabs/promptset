from io import BytesIO
from uuid import uuid4
from fastapi import Request, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.llm import QuestionAnsweringLLm
from .voice.speech_to_text import handle_speech_to_text
from .voice.text_to_speech import handle_text_to_speech


async def handle_text_answer(request: Request):
    file_name = f"{uuid4()}.mp3"
    audio_data = await request.body()
    file = UploadFile(filename=file_name, file=BytesIO(audio_data))
    text = await handle_speech_to_text(file)
    llm = QuestionAnsweringLLm.get_model()
    messages = [
        SystemMessage(
            content="You're a helpful AI assistant named `Voice`, You help people with their general purpose queries. If you cannot answer or perform ceratin tasks, you just reply with `I'm sorry, I cannot do that yet.`"
        ),
        HumanMessage(content=text),
    ]

    aiMessage = await llm.ainvoke(messages)
    return aiMessage.content


async def handle_audio_answer(request: Request):
    text = await handle_text_answer(request)
    response = await handle_text_to_speech(text)
    return response
