from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import base64, json, asyncio

from ..utils import sse_format, conv_h_to_str
from services import openai_service


router = APIRouter()

@router.get("/getcourse", tags=["Course"], summary="Generate course content based on provided outline and topic.")
async def get_course(outline: str, topic: str):
    """
    Based on a provided outline and topic, this endpoint generates course content by communicating with the OpenAI API in an asynchronous manner.
    """
    # Decode and load the outline
    decoded_outline = base64.b64decode(outline).decode('utf-8')
    outline_data = json.loads(decoded_outline)
    outline_str = conv_h_to_str(outline_data['items'])
    tasks = []

    for index, item in enumerate(outline_data['items']):
        for s_index, subtopic in enumerate(item['subtopics']):
            task = openai_service.create_subtopic_task(index, s_index, topic, outline_str)
            tasks.append(task)

    async def event_stream():
        for task in asyncio.as_completed(tasks):
            result = await task
            yield sse_format(result)

    return StreamingResponse(event_stream(), media_type="text/event-stream")