import asyncio
import en_core_web_sm
from langchain.text_splitter import SpacyTextSplitter
from db.segment import new_seg
from utils.azure_open_ai import get_chat_completion
import json
from config import CHECK_CHUNKS
import logging

# Get the logger for the module
logger = logging.getLogger('langchain.text_splitter')

# Set the level to ERROR
logger.setLevel(logging.ERROR)

# function to test if segment is worth keeping or not


async def seg_checker(seg):
    messages = [
        {'role': "system", 'content': 'LLM is an expert evaluator of text. It can determine whether a piece of text is substantial or just gibberish. It carefully considers the text and whether a human would find any substantial meaning from it.'},
        {"role": "user", "content": seg}
    ]

    functions = [
        {
            "name": "text_checker",
            "description": "Your role is to determine whether the user message is substantial text or just gibberish. In determining whether it is substantial, it should be more than just nonsenstical characters and should include full sentences and not just document headers or footers. If it is just a URL, consider it gibberish. If you are unsure, err on the side of considering it substantial.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meaningful": {
                        "type": "boolean",
                        "description": "return true if the text is substantial or you are unsure, return false if the text is gibberish"
                    },

                },
                "required": ["meaningful"]
            }
        }
    ]

    try:
        check = await get_chat_completion(messages, functions=functions, function_to_call={"name": "text_checker"})

        return json.loads(check["message"]["function_call"]["arguments"])["meaningful"]
    except Exception as e:
        return True


def chunker(text):

    text_splitter = SpacyTextSplitter(chunk_size=450, chunk_overlap=30)

    texts = text_splitter.split_text(text)

    return texts


async def segs_processor(text, parent_id):
    segments = chunker(text)

    # runs segment processing in parallel to speed up
    async def process(seg):
        if CHECK_CHUNKS:
            test = await seg_checker(seg)
        else:
            test = True
        if test:
            seg = {
                "text": seg,
            }
            return await new_seg(seg, parent_id)

    tasks = [process(seg) for seg in segments]
    seg_ids = await asyncio.gather(*tasks)
    # remove None values
    seg_ids = [seg_id for seg_id in seg_ids if seg_id is not None]

    return seg_ids
