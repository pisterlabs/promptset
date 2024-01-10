from typing import Any

import openai

from .constants import FT_PREDICTOR_MODEL, OPENAI_API_KEY, PREDICTOR_SYSTEM_PROMPT

openai.api_key = OPENAI_API_KEY


async def predict_functions(prompt: str) -> list[str]:
    return []
    messages = [
        {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    predict_functions_result: Any = await openai.ChatCompletion.acreate(  # type: ignore
        model=FT_PREDICTOR_MODEL, messages=messages, temperature=0, max_tokens=500
    )

    predict_function = predict_functions_result["choices"][0]["message"]

    function_names = predict_function["content"].split(", ")
    return [function_name.replace(".", "_") for function_name in function_names]
