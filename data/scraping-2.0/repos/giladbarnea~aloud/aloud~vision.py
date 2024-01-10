import textwrap
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

from openai import BadRequestError
from plum import dispatch

from aloud.console import console
from aloud.openai import oai
from aloud.text import (
    LINE_NUMBER_SEPARATOR,
    add_line_numbers,
    has_line_numbers,
    remove_line_numbers,
)


@console.with_status('Generating and injecting image descriptions...')
def inject_image_descriptions_as_alt(markdown: str) -> str:
    markdown_with_line_numbers = add_line_numbers(markdown)
    image_link_indices: list[int] = get_image_link_indices(markdown_with_line_numbers)
    image_links: list[str] = extract_image_links(markdown_with_line_numbers, image_link_indices)
    image_descriptions: list[str] = generate_image_descriptions(image_links)
    markdown = remove_line_numbers(markdown_with_line_numbers)
    markdown_lines = markdown.splitlines()
    line_index: int
    image_link: str
    image_description: str
    for line_index, image_link, image_description in zip(image_link_indices, image_links, image_descriptions):
        line = markdown_lines[line_index]
        src_attribute_jindex = line.index('src=')
        line = line[:src_attribute_jindex] + f'alt="{image_description}" ' + line[src_attribute_jindex:]
        markdown_lines[line_index] = line
    markdown = '\n'.join(markdown_lines)
    assert not has_line_numbers(markdown)
    return markdown


def generate_image_descriptions(image_links: list[str]) -> list[str]:
    image_description_futures = []
    with ThreadPoolExecutor(max_workers=len(image_links)) as executor:
        for image_link in image_links:
            future = executor.submit(generate_image_description, image_link)
            image_description_futures.append(future)
    image_descriptions = [future.result() for future in image_description_futures]
    return image_descriptions


def generate_image_description(image_link: str) -> str:
    prompt = (
        textwrap.dedent(
            """
            You are given an image link.
            Generate a short description for the image, and return exactly it, without explanation or anything else.
            If the image link is broken, or the image is not accessible, or the image is not an image, return: None
            
            The image link:
            {image_link}
            """,
        )
        .format(image_link=image_link.strip())
        .strip()
    )
    try:
        chat_completion = oai.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_link, 'detail': 'high'}},
                    ],
                }
            ],
            model='gpt-4-vision-preview',
            temperature=0,
            stream=False,
            timeout=10,
            max_tokens=1000,
        )
    except BadRequestError as e:
        console.log(f'⚠️ {type(e).__name__}: {e} │ {image_link}')
        return ''

    image_description = chat_completion.choices[0].message.content.splitlines()[0].strip()
    return image_description


@dispatch
def extract_image_links(markdown: str) -> list[str]:
    image_link_indices = get_image_link_indices(markdown)
    return extract_image_links(markdown, image_link_indices)


@dispatch
def extract_image_links(markdown: str, image_link_indices: list[int]) -> list[str]:
    image_link_futures = []
    with ThreadPoolExecutor(max_workers=len(image_link_indices)) as executor:
        markdown_lines = markdown.splitlines()
        for index in image_link_indices:
            line = markdown_lines[index]
            future = executor.submit(extract_image_link, line)
            image_link_futures.append(future)
    image_links = [future.result() for future in image_link_futures]
    return image_links


def get_image_link_indices(markdown: str) -> list[int]:
    assert has_line_numbers(markdown)
    prompt = (
        textwrap.dedent(
            f"""
            You are given a markdown representation of an article from the internet, generated automatically by a tool. This means that the markdown is not perfect.
            Line numbers are shown on the left "gutter" of the markdown; the special vertical line `{LINE_NUMBER_SEPARATOR}` separates each line number from the rest of the line, and the first line is 0.
            Find all the image links in the article, and return their line numbers as listed in the gutters of the lines where the image links appear, separated by line breaks, and only them, without explanation or anything else.
            If the article does not contain any images, return: None

            The article's markdown representation is:

            {{markdown}}

            """,
        )
        .format(markdown=markdown.strip())
        .strip()
    )
    chat_completion = oai.chat.completions.create(
        messages=[{'role': 'system', 'content': prompt}],
        model='gpt-4-1106-preview',
        temperature=0,
        stream=False,
        timeout=10,
    )
    image_link_indices = []
    for index in chat_completion.choices[0].message.content.splitlines():
        image_link_indices.append(int(index.strip()))
    return image_link_indices


def extract_image_link(markdown_line: str) -> str:
    prompt = (
        textwrap.dedent(
            """
            You are given a markdown line which contains an image link.
            Extract the image link, and return exactly it, without explanation or anything else.
            Note that sometimes, the domain of the link is that of a CDN, or analytics, or similar, but the actual image link is a parameter of the URL. An oversimplified example of such a link is: https://cdn.irrelevant.com/https://actual-link.com/image.png?width=100&height=100, in which case, you would return: https://actual-link.com/image.png 

            The markdown line:
            {markdown_line}

            """,
        )
        .format(markdown_line=markdown_line.strip())
        .strip()
    )
    chat_completion = oai.chat.completions.create(
        messages=[{'role': 'system', 'content': prompt}],
        model='gpt-4-1106-preview',
        temperature=0,
        stream=False,
        timeout=10,
    )
    image_link = chat_completion.choices[0].message.content.splitlines()[0].strip()
    unquoted_image_link = unquote(image_link)
    return unquoted_image_link
