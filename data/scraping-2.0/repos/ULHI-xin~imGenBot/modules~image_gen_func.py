import logging

import colorama
import openai


def gen_image_from_prompt(
    prompt: str,
    image_size: str,
    n: int
): 
    logging.info(f"gen_image:" + colorama.Fore.BLUE +
                 f"prompt={prompt!r}, size={image_size!r}, n={n!r}" +
                 colorama.Style.RESET_ALL)

    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=image_size
    )

    logging.info(f"gen_image:" + colorama.Fore.GREEN +
                 f"resp={response}" + colorama.Style.RESET_ALL)

    result_urls = []
    for i in response['data']:
        result_urls.append(i['url'])

    return result_urls
