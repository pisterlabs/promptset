import openai
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import json
from web3 import Web3


class LockAIResponse(BaseModel):
    owner: Optional[str]
    unlockTime: Optional[str]


lock_schema = LockAIResponse.model_json_schema()
lock_null_object = LockAIResponse(owner=None, unlockTime=None).model_dump_json()


def validate_lock_input(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "get_answer_for_user_query",
                "description": "Ask user for time-lock parameters, unknown parameters should be null",
                "parameters": LockAIResponse.model_json_schema(),
            }
        ],
        function_call={"name": "get_answer_for_user_query"},
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    date_string = output.get("unlockTime")
    if date_string is not None:
        try:
            output["unlockTime"] = int(
                datetime.strptime(date_string, "%Y-%m-%d %H:%M").timestamp()
            )
        except:
            # should also handle conversion to UTC will assume UTC for demo
            return False, {
                "role": "assistant",
                "content": "To avoid inconsistencies with date formats please use the date format YYYY-MM-DD HH:MM for the unlock time",
            }
            # return False, "To avoid inconsistencies with date formats please use the date format YYYY-MM-DD HH:MM for the unlock time"

    if output.get("owner") is not None:
        try:
            output["owner"] = Web3.toChecksumAddress(output["owner"])
        except:
            output["owner"] = None
            return False, {
                "role": "assistant",
                "content": "The owner you provided is not a valid address, can you please provide the owners address?"
            }

    return True, output
