#!/usr/bin/env python
import openai

import util

def main():
    prompt = util.prompt_prompt()
    size   = util.prompt_image_size()
    number = util.prompt_number()
    openai.api_key = util.prompt_api_key()

    resp = openai.Image.create(
        prompt=prompt,
        n=number,
        size=size)

    util.confirm(f'''\
prompt = {prompt}
size   = {size}
number = {number}
''')
    
    timestamp = resp.created
    urls = [ res['url'] for res in resp.data ]
    try:
        util.download_to_directory(urls, timestamp)
    except Exception as e:
        print(resp)
        print(e)

if __name__ == '__main__':
    main()

