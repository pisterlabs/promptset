import asyncio
from collections import defaultdict
from typing import Any

import pandas as pd
from langchain.output_parsers import ResponseSchema
from malevich.square import DF, Context, processor

from ..lib.broadcast import broadcast
from ..lib.chat import exec_structured_chat


@processor(id="structured_prompt_completion")
async def structured_prompt_completion(variables: DF[Any], ctx: Context):
    """Use Chat Completions feature from OpenAI

    Chat completions enable you to chat with OpenAI
    using a prompt template. This processor assures the
    output to follow a certain schema

    Available models: https://platform.openai.com/docs/models

    To use the model you should set the following parameters:

    - `openai_api_key`: Your OpenAI API key. Get it here: https://platform.openai.com/api-keys
    - `user_prompt`: The prompt for the user

    Scroll down to see the full list of parameters.

    Inputs:

        A dataframe with variables to be used in the prompts. Each row of the
        dataframe will be used to generate a prompt. For example, if your prompt
        contains a name enclosed in {} like this:

        Hi! Write a story about {someone}

        You have to have a column `someone` in the input dataframe. For each
        of such variables you should have a separate column.

    Outputs:

        A dataframe with following column:
        - index (int): the index of the variable. If `include_index` is set to
            True, the index will be included in the output, otherwise it will
            be omitted.
        - content (str): the content of the model response

    Configuration:

        - openai_api_key (str, required): your OpenAI API key
        - user_prompt (str, required): the prompt for the user
        - model (str, default: 'gpt-3.5-turbo'): the model to use
        - organization (str, default: None): the organization to use
        - max_retries (int, default: 3): the maximum number of retries
        - temperature (float, default: 0.9): the temperature
        - max_tokens (int, default: 150): the maximum number of tokens
        - top_p (float, default: 1.0): the top p
        - frequency_penalty (float, default: 0.0): the frequency penalty
        - presence_penalty (float, default: 0.0): the presence penalty
        - stop (list, default: []]): the stop tokens
        - stream (bool, default: False): whether to stream the response
        - n (int, default: 1): the number of completions to generate
        - response_format (str, default: None): the response format
        - fields (list of dict, default: empty): a list of fields to parse the output.
            Each field is a dict that contains fields `name`, `description` and `type`
        - include_index (bool, default: False): whether to include the index in the output

    Notes:
        If `response_format` is set to 'json_object', the system prompt should
        contain an instruction to return a JSON object, e.g.:

        ```
        You are a creative writer. You are writing a story about {names}.
        You should mention {colors} in your story.

        JSON:
        ```

        JSON completion only works with Davinci models

    Examples:
        user_prompt: "Write a search queries to find {something} on the Internet"
        fields: [
            {"name": "query", "description": "Google search query", "type": "List[string]"},
            {"name": "description", "description": "Summary of your result", "type": "string"}
        ]

        If you have the config like above, you have to expect the following output:

        | index |    query   | description |
        | ----- | ---------- | ----------- |
        | 0     |  query_1   |   summary   |
        | 0     |  query_2   |   summary   |

        You will see repeated index and values if some of the variables are lists and
        others are not. This is because the output is a cartesian product of all the
        variables.

    Args:
        variables (DF[Any]): the variables to use in the prompts
        ctx (Context): the context

    Returns:
        DF[Any]: the chat messages
    """  # noqa: E501

    try:
        conf = ctx.app_cfg["conf"]
    except KeyError:
        raise Exception("OpenAI client not initialized.")

    assert "user_prompt" in ctx.app_cfg, "Missing `user_prompt` in app config."

    # system_prompt = ctx.app_cfg.get('system_prompt', '')
    user_prompt = ctx.app_cfg["user_prompt"]

    messages = [
        user_prompt.format(**_vars) for _vars in variables.to_dict(orient="records")
    ]

    schema = [
        ResponseSchema(
            name=field["name"],
            description=field["description"],
            type=field.get("type", "string"),
        )
        for field in ctx.app_cfg.get("fields", [{}])
    ]

    response = await asyncio.gather(
        *[exec_structured_chat(message, conf, schema) for message in messages]
    )

    df = defaultdict(lambda: [])

    for i, message in enumerate(response):
        ln = 0
        for key, value in broadcast(message).items():
            df[key].extend(value)
            ln = len(value)
        if conf.include_index:
            df["index"].extend([variables.index[i]] * ln)

    return pd.DataFrame(df)
