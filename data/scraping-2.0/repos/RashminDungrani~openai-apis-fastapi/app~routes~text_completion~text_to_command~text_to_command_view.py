from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.core.exceptions import DetailedHTTPException
from app.dependencies import openai_api_handle

router = APIRouter()


from fastapi import APIRouter, Depends, Query

router = APIRouter()


@router.get("/v1")
async def text_to_command_v1(query: str = Query(min_length=3)):
    #     completions = openai.Completion.create(
    #     engine="davinci",
    #     prompt=prompt,
    #     max_tokens=60,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )
    response = openai_api_handle(
        model="text-davinci-003",
        prompt=f"Convert this text to a programmatic command:\n\nExample: Ask Constance if we need some bread\n\nOutput: send-msg `find constance` Do we need some bread?\n\n{query}",
        api_end_point="/api/text_completion/text_to_command/v1",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0,
        # stop=["\n"],
    )
    if response.openai_response:
        answer = response.openai_response.choices[0].text.strip()
        return {"answer": answer}

    raise DetailedHTTPException()
