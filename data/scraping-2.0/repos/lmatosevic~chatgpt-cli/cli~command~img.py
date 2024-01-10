import os
import sys
from urllib.request import urlopen

import openai

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cli.core import ensure_api_key, read_stdin, valid_api_key, valid_input, image_url_response


def run():
    """
    gpt-img [api_key] [prompt] [img_out]
    """

    content = read_stdin()

    key_in_args = False
    if len(sys.argv) > 3:
        img_out = str(sys.argv[3])
        prompt = str(sys.argv[2])
        key_in_args = True
    elif len(sys.argv) > 2:
        img_out = str(sys.argv[2])
        prompt = str(sys.argv[1])
        if valid_api_key(prompt):
            key_in_args = True
            prompt = None
            if content is None:
                prompt = img_out
                img_out = None
    elif len(sys.argv) > 1:
        img_out = str(sys.argv[1])
        prompt = None
        if valid_api_key(img_out):
            key_in_args = True
            img_out = None
            prompt = None
        elif content is None:
            prompt = img_out
            img_out = None
    else:
        img_out = None
        prompt = None

    if not valid_input(prompt) and not valid_input(content):
        print('No input provided by either stdin nor command argument. '
              'Usage example: cat description.txt | gpt-img "with cartoon graphics" out.png')
        sys.exit(1)

    openai.api_key = ensure_api_key(prompt=True, use_args_key=key_in_args)

    conbined_prompt = ''
    if valid_input(content):
        conbined_prompt = conbined_prompt + content
    if valid_input(prompt):
        if len(conbined_prompt) > 0:
            conbined_prompt = conbined_prompt + '. '
        conbined_prompt = conbined_prompt + prompt

    response = image_url_response(conbined_prompt)
    if response is None:
        sys.exit(2)

    img = urlopen(response).read()

    if img_out is None:
        stdout = os.fdopen(sys.stdout.fileno(), "wb", closefd=False)
        stdout.write(img)
        stdout.flush()
    else:
        image_file = open(img_out, 'wb')
        image_file.write(img)
        image_file.close()


if __name__ == '__main__':
    run()
