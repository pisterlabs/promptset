import json
from datetime import datetime

import openai
import ujson
from fastapi import HTTPException

from app.app_paths import app_paths
from app.core.exceptions import DetailedHTTPException
from app.core.settings import settings
from app.models.models import APIResult, TextCompletion


def get_openai():
    # sets open API API KEY
    openai.api_key = settings.OPENAI_API_KEY
    return openai


def openai_api_handle(
    model: str,
    prompt: str,
    api_end_point: str,
    save_record_to_json=True,
    temperature: float = 0,
    max_tokens=100,
    top_p: float = 1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop: list[str] | None = None,
) -> APIResult:
    try:
        response = get_openai().Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        try:
            text_completion = TextCompletion.parse_obj(response)

            # Write the list of TextCompletion objects to a JSON file
            api_result = APIResult(
                api_end_point=api_end_point,
                input=prompt,
                called_at=datetime.now(),
                openai_response=text_completion,
            )
            if save_record_to_json:
                # Open file in r+ mode to read and append
                with open(app_paths.openai_responses_json, "r+", encoding="utf-8") as f:
                    # Load JSON data from file
                    data = json.load(f)

                    # Move the file pointer to the beginning of the file
                    f.seek(0)

                    # Append new data to the list
                    data.append(ujson.loads(api_result.json()))

                    # Write the updated JSON data back to the file with indentation
                    json.dump(data, f, indent=4, ensure_ascii=False)

                    # Truncate the file to the current position
                    f.truncate()

            return api_result

        except Exception as e:
            print("***** Exception *****")
            print(e)
            raise DetailedHTTPException()
    except openai.APIError as error:
        raise HTTPException(status_code=error.http_status, detail=error.json_body)  # type: ignore
