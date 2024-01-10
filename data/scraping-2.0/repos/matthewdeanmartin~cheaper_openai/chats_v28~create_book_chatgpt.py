import time

import marko
import openai
import yaml

from chats.ai_utils.client_utils import create_client
from chats.ai_utils.io_utils import dump_response, read_config, read_prompt, read_yaml_toc_prompt
from chats.ai_utils.token_utils import count_tokens

config = read_config()
create_client()

WHO_ARE_YOU = """
You are the author of python scripting tutorial books.
You want to teach the world how to do everything that otherwise would be done with bash, but instead you will show them how to do it with python.
"""
TITLE = "Filesystem scripting with python with examples inspired by Dr Doolittle"
TEMPLATE = f"""Create a table of contents for a 25 chapter book named '{TITLE}' The output must be in yaml. 
In this format

```yaml
---
- Chapter
  - Section
  - Another Section
- Another Chapter
  - Another Section
  - Another Section
```

```yaml
---
- Reading from the file system
  - What is a file system
  - Relevant APIs
  - Common tasks with code examples
- Reading to the file system
  - Relevant APIs
  - Common tasks with code examples
```
"""


def fulfill_the_promise():
    # get output folder from config file
    # output_folder = config["output"]["output_folder"]
    # prompt = read_prompt(output_folder)

    markdown_document = """
    # title

    ## subtitle first

    paragraph

    ## subtitle last

    paragraph one.

    paragraph two.
    """

    tree = marko.parse(markdown_document)

    # Find the last subtitle node in the tree
    last_subtitle = None
    for node in tree.children:
        if node.tag_name == "h2":
            last_subtitle = node

    # Extract the markdown under the last subtitle
    markdown_text = ""
    for node in last_subtitle.children:
        markdown_text += node.markdown

    return markdown_text


def make_the_toc() -> None:
    # get output folder from config file
    output_folder = config["output"]["output_folder"]

    prompt = read_prompt(output_folder)

    temperature = 0.9
    max_tokens = 4000
    response = basic_request(max_tokens=max_tokens, prompt=TEMPLATE, who_are_you=WHO_ARE_YOU, temperature=temperature)

    choices = choices_from_response(response)
    dump_response(f"{WHO_ARE_YOU} \n\n {prompt}", choices, "book.yml", output_folder)

    likely_yaml = "".join(choices)
    yaml.safe_load(likely_yaml)


def choices_from_response(response):
    return list(x["message"]["content"] for x in response["choices"])


def run_the_toc():
    temperature = 0.9
    max_tokens = 4000

    # get output folder from config file
    output_folder = config["output"]["output_folder"]
    toc = read_yaml_toc_prompt(output_folder)
    section_count = 0

    for inner in toc:
        for section, chapters in inner.items():
            section_count += 1
            chapter_count = 0
            for chapter in chapters:
                chapter_count += 1

                prompt = f"Please write the exposition for '{section} : {chapter}'. Use markdown."
                response = basic_request(max_tokens, prompt, WHO_ARE_YOU, temperature, sleep=5)
                choices = choices_from_response(response)
                dump_response(f"{section} : {chapter}", choices, f"{section}_{chapter_count}", output_folder)

                prompt = (
                    f"Please write code samples for '{section} : {chapter}'. "
                    f"Use markdown and code blocks for code. All examples *must* use characters, scenes, quotes from the Doctor Dolittle series by Hugh Lofting as example material. Be creative and playful."
                )
                response = basic_request(max_tokens, WHO_ARE_YOU, prompt, temperature, sleep=5)
                choices = choices_from_response(response)
                dump_response(f"{section} : {chapter}", choices, f"{section}_{chapter_count}_examples", output_folder)


def basic_request(max_tokens, prompt, who_are_you, temperature, sleep=0):
    prompt_tokens = count_tokens(prompt)
    print(prompt)
    time.sleep(sleep)
    messages = [
        {"role": "system", "content": who_are_you},
        {"role": "user", "content": prompt},
        # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        # {"role": "user", "content": "Where was it played?"}
    ]
    args = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens - prompt_tokens,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    response = openai.ChatCompletion.create(**args)
    return response


if __name__ == "__main__":
    # make_the_toc()
    run_the_toc()
