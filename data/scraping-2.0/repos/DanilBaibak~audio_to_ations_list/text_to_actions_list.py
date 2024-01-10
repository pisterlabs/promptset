import re
import logging.config

import openai
import tiktoken

from scripts.utils import read_from_file, get_output_file_path, arg_parser, parse_config

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

AI_MODEL = "text-davinci-003"
MAX_TOKENS_NUM = 4000  # our estimate is approximate, so we will use a slightly lower maximum number
REQUEST_TEMPLATE = "--Full meeting transcript-- \n {} \n\n --Action items bullet list--"


def get_action_items_for_text(prompt: str) -> str:
    response = openai.Completion.create(
        engine=AI_MODEL,
        prompt=prompt,
        max_tokens=1500,
        n=1,
        temperature=0,
    )

    if response["choices"][0]["finish_reason"] == "error":
        logging.error(f"Error: {response['choices'][0]['text']}")
        exit(1)

    return response.choices[0].text


def get_num_of_tokens(text: str, model="text-davinci-003") -> int:
    """ Get the approximate number of tokens used by the text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoding.encode(text))

    return num_tokens


def main():
    # get config data
    config = parse_config("config.ini")
    openai.api_key = config.get("api_keys", "OPENAI_KEY")

    args = arg_parser()

    # prepare the output path
    input_text = read_from_file(args.input_file_path)
    output_file_path = get_output_file_path(args.input_file_path, args.output_file_path)

    # estimate num of tokens in the text
    num_of_tokens = get_num_of_tokens(input_text)
    logging.info(f"Total tokens number: {num_of_tokens}")

    # get action items
    if num_of_tokens < MAX_TOKENS_NUM:
        input_text = REQUEST_TEMPLATE.format(input_text)
        action_items = [get_action_items_for_text(input_text)]
    else:
        num_of_chunks = num_of_tokens // MAX_TOKENS_NUM + 1
        max_chunk_size = num_of_tokens // num_of_chunks
        logging.info(f"Number of chunks: {num_of_chunks}")
        logging.info(f"Max chunk size: {max_chunk_size}")

        sentences = re.split("(?<=[.!?])\s+", input_text)
        chunk_size, chunk_text = 0, ''
        action_items = []

        # split the text into number of chunks to fit the API limit.
        for sentence in sentences:
            chunk_size += get_num_of_tokens(sentence)
            chunk_text += ' ' + sentence

            if chunk_size >= max_chunk_size:
                # get action items per chunk
                input_text = REQUEST_TEMPLATE.format(chunk_text)
                actions = get_action_items_for_text(input_text)

                action_items.append(actions)
                chunk_size, chunk_text = 0, ''

        if len(chunk_text) > 0:
            input_text = REQUEST_TEMPLATE.format(chunk_text)
            actions = get_action_items_for_text(input_text)
            action_items.append(actions)

    # save the output
    with open(output_file_path, "w") as output_file:
        output_file.write("".join(action_items))

    logging.info(f"The list of actions has been saved to the file - {output_file_path}")


if __name__ == "__main__":
    main()
