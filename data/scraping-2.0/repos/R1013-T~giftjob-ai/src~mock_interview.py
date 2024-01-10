import os
import json
from openai import AzureOpenAI
from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

deployment_name = 'entry-sheet'


async def MockInterview(occupation: str, user: Optional[List[str]] = None, question: Optional[List[str]] = None, previous_messages: Optional[List[Message]] = None, isFirst: Optional[bool] = False, isLast: Optional[bool] = False, isEnd: Optional[bool] = False):
    message = [
        {"role": "system", "content": "これから就職活動における、面接を行ってください。"},
        {"role": "system", "content": f"職種は、{occupation}です。"},
    ]

    if isFirst == True:
        message.append(
            {"role": "system",
                "content": '''まず、最初の挨拶と、質問をしてください。
                出力形式はJSONで、以下の形式で出力してください。
                "greeting": 挨拶の内容
                "question": 質問の内容
                '''},
        )
    elif isLast == False and isEnd == False:
        message = previous_messages
        message.append(
            {"role": "system",
                "content": '''これまでの質疑応答を踏まえて、次の質問をしてください。
                出力形式はJSONで、以下の形式で出力してください。
                "question": 質問の内容
                '''},
        )
    elif isLast == True:
        message = previous_messages
        message.append(
            {"role": "system",
                "content": '''これまでの質疑応答を踏まえて、最後の質問をしてください。最後の質問ということも伝えてください。
                出力形式はJSONで、以下の形式で出力してください。
                "question": 質問の内容
                '''},
        )
    elif isEnd == True:
        message = previous_messages
        message.append(
            {"role": "system",
                "content": '''これで面接は終了です。質問は終わりの旨を伝え、最後の挨拶とアドバイスを出力してください。
                出力形式はJSONで、以下の形式で出力してください。
                "greeting": 質問は終わり、挨拶の内容
                "advice": アドバイスの内容
                "score": 面接の総評点を100点満点で出力してください。
                '''},
        )

    response = client.chat.completions.create(
        model="entry-sheet",
        response_format={"type": "json_object"},
        messages=message,
    )

    response_content = response.choices[0].message.content

    data = json.loads(response_content)

    res = {
        "greeting": data.get("greeting", None),
        "question": data.get("question", None),
        "advice": data.get("advice", None),
        "score": data.get("score", None)
    }

    return {"occupation": occupation, "previous_messages": previous_messages, "message": message, "isFirst": isFirst, "isLast": isLast, "isEnd": isEnd, "res": res}
