import json
import logging
import openai
from fastapi import HTTPException
from openai.error import Timeout

logger = logging.getLogger(__name__)

message = "查看访客"

functions = [
    {
        "name": "navigation",
        "description": "导航至某个页面",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "页面: 轨迹大屏, 数据大屏, 访客列表",
                    "enum": ["data", "track", "visitor"],
                }
            },
            "required": ["type"],
        },
    },
]
max_retries = 2
retry_count = 0
while True:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": message}],
            functions=functions,
            function_call={"name": "navigation"},
            request_timeout=2,
        )
        data = response["choices"][0]["message"]["function_call"]
        print(data)
        break
    except Timeout:
        retry_count += 1
        if retry_count > max_retries:
            raise HTTPException(
                status_code=504,
                detail="???",
            )
        else:
            logger.warning("???")
            continue
