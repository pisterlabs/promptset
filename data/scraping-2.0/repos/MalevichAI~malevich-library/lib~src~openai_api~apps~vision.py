import asyncio
import os
from typing import Any

import pandas as pd
from malevich.square import DF, Context, processor

from ..lib.vision import exec_vision
from ..models.configuration.base import Configuration


@processor()
async def completion_with_vision(variables: DF[Any], ctx: Context):
    """Use Language Model with Vision feature from OpenAI

    Completion with Vision enables you to generate text
    based on an image.

    The model is set to 'gpt-4-vision-preview' by default.

    Configuration:

        - openai_api_key (str, required): your OpenAI API key
        - user_prompt (str, required): the prompt for the user
        - image_column (str, default: 'images'): the column with images
        - max_tokens (int, default: 2048): the maximum number of tokens
        - top_p (float, default: 1.0): the probability of the model
            returning a next token that is in the top P tokens
        - temperature (float, default: 1.0): the higher the value,
            the more random the generated text
        - frequency_penalty (float, default: 0.0): the higher the value,
            the less likely the model is to repeat the same word
        - presence_penalty (float, default: 0.0): the higher the value,
            the less likely the model is to talk about the same topic again
        - model (str, default: 'gpt-4-vision-preview'): the model to use


    Inputs:

        A dataframe with variables to be used in the prompts and an extra
        column with images files or urls.

        A column {image_column} should contain either a path to the image
        file or a url to the image.

        Each row of the dataframe will be used to generate a prompt.
        For example, if your prompt contains a name enclosed in {} like this:

        Hi! Write a story about {someone}

        You have to have a column `someone` in the input dataframe. For each
        of such variables you should have a separate column.

    Outputs:

        A dataframe with following columns:
            - content (str): the content of the model response

    Supported file types:

        - png
        - jpg
        - jpeg
        - gif
        - bmp
        - tiff
        - tif
        - webp

    Args:
        variables (DF[Any]): a dataframe with variables
        ctx (Context): context

    Returns:
        A dataframe with model responses
    """


    image_column = ctx.app_cfg.get("image_column", "images")
    assert image_column in variables.columns, f"Missing `{image_column}` column."

    try:
        conf: Configuration = ctx.app_cfg["conf"]
    except KeyError:
        raise Exception("OpenAI client not initialized.")

    if not conf.model:
        conf.model = "gpt-4-vision-preview"

    assert "user_prompt" in ctx.app_cfg, "Missing `user_prompt` in app config."

    system_prompt = ctx.app_cfg.get("system_prompt", "")
    user_prompt = ctx.app_cfg["user_prompt"]

    def __is_file(x) -> bool:
        return os.path.isfile(
            ctx.get_share_path(x, not_exist_ok=True)
        ) or os.path.isfile(x)

    messages = [
        [
            {
                "role": "system",
                "content": system_prompt.format(**_vars)
            },
            {
                "role": "user",
                "content": user_prompt.format(**_vars)},
        ]
        for _vars in variables.to_dict(orient="records")
    ]

    images = variables[image_column].tolist()

    for i in range(len(images)):
        if __is_file(images[i]):
            images[i] = ctx.get_share_path(images[i])

    response = await asyncio.gather(
        *[exec_vision(msgs, image, conf, not __is_file(image))
          for msgs, image in zip(messages, images)]
    )

    df = {
        "content": [],
    }

    for _response in response:
        for _message in _response:
            df["content"].append(_message.content)

    return pd.DataFrame(df)
