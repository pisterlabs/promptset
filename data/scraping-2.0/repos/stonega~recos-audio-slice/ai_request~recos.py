"""
Code from https://github.com/rongjc/autosubtitle/blob/main/translate.py
"""
import ast
import time
import openai
from utils import logger
from ai_request.utils import group_chunks, num_tokens_from_messages

def get_recos(text):

    prompt_text = f"""
I want you to extract info from text if any movies books you found.Return the item list split with |,return result with json object, Text:
{text}"""
    print(prompt_text)
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        t_text = (
            completion["choices"][0]  # type: ignore
            .get("message")
            .get("content")
            .encode("utf8")
            .decode()
        )
        # format the translated text, the original text is eg: "\n\n['\\n柠檬\\n\\n', '梶井基次郎']", we need the
        # element in the list, not the \n \n

        try:
            t_text = ast.literal_eval(t_text)
        except Exception:
            # some ["\n"] not literal_eval, not influence the result
            pass
        # openai has a time limit for api  Limit: 20 / min
        time.sleep(3)
    except Exception as e:
        logger.info(str(e), "will sleep 60 seconds")
        # TIME LIMIT for open api please pay
        time.sleep(60)
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        t_text = (
            completion["choices"][0]  # type: ignore
            .get("message")
            .get("content")
            .encode("utf8")
            .decode()
        )
        t_text = t_text.strip("\n")
        try:
            t_text = ast.literal_eval(t_text)
        except Exception:
            pass
    logger.info(t_text)
    return t_text


def subtitle_recos(subtitles):
    ntokens = []
    chunks = []
    for subtitle in subtitles:
        chunk = subtitle['text']
        chunks.append(chunk)
        ntokens.append(num_tokens_from_messages(chunk))

    chunks = group_chunks(chunks, ntokens)
    recos_chunks = {
        "books": [],
        "movies": [],
    }
    for i, chunk in enumerate(chunks):
        print(str(i+1) + " / " + str(len(chunks)))
        result = get_recos(chunk)
        

    result = "\n".join(recos_chunks)

    return result
