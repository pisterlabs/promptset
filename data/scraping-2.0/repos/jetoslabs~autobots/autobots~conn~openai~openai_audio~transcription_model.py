from typing import Literal

# from openai._types import FileTypes
from pydantic import BaseModel, HttpUrl


class TranscriptionReq(BaseModel):
    file_url: HttpUrl | None = None
    # file: FileTypes
    model: Literal["whisper-1"] = "whisper-1"
    language: str = "en"
    prompt: str = ""
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json"
    temperature: float = 0.2
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    # extra_headers: Headers | None = None,
    # extra_query: Query | None = None,
    # extra_body: Body | None = None,
    # timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,