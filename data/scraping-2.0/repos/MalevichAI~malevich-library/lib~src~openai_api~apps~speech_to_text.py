import asyncio
import os
import shutil
import uuid
from typing import Any

import pandas as pd
from malevich.square import DF, Context, processor

from ..lib.whisper import exec_whisper


@processor()
async def speech_to_text(variables: DF[Any], ctx: Context):
    """Use Speech-to-Text feature from OpenAI

   Transcribe audio to text using OpenAI API

    To use the model you should set the following parameters:

    - `openai_api_key`: Your OpenAI API key. Get it here: https://platform.openai.com/api-keys

    Scroll down to see the full list of parameters.

    Inputs:

        A dataframe with a single column `filename` containing keys
        to shared audio files

    Outputs:

        A dataframe with following columns:
            - content (str): audio transcription

    Configuration:

        - openai_api_key (str, required): your OpenAI API key
        - language (str, default: 'en'): the language to transcribe
            audio to
        - temperature (float, default: 0.9): the temperature
        - prompt (str, default: None): a short optional prompt to
            guide the model

    Args:
        variables (DF[Any]): the variables to use in the prompts
        ctx (Context): the context

    Returns:
        DF[Any]: the chat messages
    """

    try:
        conf = ctx.app_cfg["conf"]
    except KeyError:
        raise Exception("OpenAI client not initialized.")

    files = []
    for f in variables.filename.to_list():
        if not os.path.exists(ctx.get_share_path(f)):
            raise Exception(f"File {f} does not exist.")
        _f = uuid.uuid4().hex
        _ext = os.path.splitext(f)[1]
        _f = f'{_f}{_ext}'
        shutil.copy(ctx.get_share_path(f), _f)
        files.append(_f)

    p = [ctx.app_cfg.get("prompt", None)] * len(variables)

    response = await asyncio.gather(
        *[exec_whisper(conf, f, p_) for f, p_ in zip(files,p)
    ])


    return pd.DataFrame(response, columns=["content"])
