import json
from typing import Tuple, Any
from datetime import datetime

from langchain.schema.messages import BaseMessageChunk, SystemMessage
from langchain.chat_models.openai import ChatOpenAI


async def rephrase(response: str, query: str) -> str:
    model = ChatOpenAI(
        modelName="gpt-3.5-turbo",
        temperature=1,
    )
    result = model.call(
        [
            SystemMessage(
                f"""
            Answer the question ultra-briefly using casual, human-friendly tone:
            ###{query}###
            and act as if you just performed this action and confirming this fact to the user, using the following response:
            ###{json.dumps(response)}###
        """
            )
        ]
    )
    content = result.content

    return content


def parseFunctionCall(result: BaseMessageChunk) -> Tuple[str, Any]:
    if result.additional_kwargs.function_call is None:
        return None
    return (
        result.additional_kwargs.function_call.name,
        json.loads(result.additional_kwargs.function_call.arguments),
    )


def currentDate() -> str:
    date = datetime.now()
    weekdays = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    weekday = weekdays[date.weekday()]
    month = str(date.month + 1).zfill(2)  # months are 1-based in Python
    day = str(date.day).zfill(2)
    year = date.year
    hours = str(date.hour).zfill(2)
    minutes = str(date.minute).zfill(2)

    return f"{weekday}, {month}/{day}/{year} {hours}:{minutes}"
