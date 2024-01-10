import openai
from fastapi import APIRouter, Header
from pydantic import BaseModel
from typing import Union

class Takeaways(BaseModel):
    ext_ref_no: str
    title: Union[str, None] = None
    transcript: str

router = APIRouter(prefix='/ai')

@router.post("/takeaways")
async def takeaways(takeaways:Takeaways, x_async_request: str = Header(None)):
    # In case if x-async-request header is true  (HTTP 202 Accepted)
    if x_async_request == "true":
        return {"id": "123",
                "ext_ref_no": takeaways.ext_ref_no,
                "title": takeaways.title,
                "transcript": takeaways.transcript,
                "status": "PENDING"}

    # In case if x-async-request header is false  (HTTP 201 Created)
    else:
        takeaway_data = ""
        chunk_size = 2048
        i = 0
        for i in range(0, len(takeaways.transcript), chunk_size):
            chunk = takeaways.transcript[i:i+chunk_size]
            if chunk[-1] == '.':
                takeaways_chunk = await call_gpt3_for_takeaways(chunk)
                takeaway_data += takeaways_chunk
                i += len(chunk)
            else:
                last_period_index = chunk.rindex('.')
                chunk = chunk[:last_period_index + 1]
                print(chunk)
            takeaways_chunk = await call_gpt3_for_takeaways(chunk)
            takeaway_data += takeaways_chunk

        data = {
            "id": "123",
            "ext_ref_no": 123,
            "title": takeaways.title,
            "transcript": takeaways.transcript,
            "takeaway_data": takeaway_data,
            "status": "SUCCESS"
        }

        return {"takeaway": data}


async def call_gpt3_for_takeaways(transcript):
    prompt = (
        f"Generate takeaways for transcript: {transcript}")
    takeaway = openai.Completion.create(engine="text-davinci-002",
                                        prompt=prompt,
                                        max_tokens=1024,
                                        n=1,
                                        stop=None,
                                        temperature=0.5
                                        )

    takeaway_result = takeaway.choices[0].text
    takeaway_result = takeaway_result.replace('\n', '\n ')

    return takeaway_result
