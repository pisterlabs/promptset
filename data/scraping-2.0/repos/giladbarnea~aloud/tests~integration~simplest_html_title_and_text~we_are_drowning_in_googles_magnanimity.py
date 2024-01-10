import textwrap
from pathlib import Path

from openai import OpenAI
from rich import print

from aloud.openai import oai

ARTICLE_URL = 'https://www.kpassa.me/posts/google'


def test_generate_audio(get_markdown):
    markdown = get_markdown(ARTICLE_URL)
    prompt = (
        textwrap.dedent(
            """
    You are given a markdown representation of an article from the internet.

    Convert the syntax of the markdown into text that can be read out. 
    The general principle is that, as you know, saying "hashtag hastag <something>" does not make sense to humans, so you should convert that to "The title of the next section is <something>.".
    Similarly, saying "Open square brackets, Press here, close square brackets, open parenthesis, https://www.google.com, close parenthesis" does not make sense to humans, so you should convert that to "There's a link to Google here.".
    Generalize this to everything that when pr6onounced literally does not make sense to humans.
    Keep the text of the article exactly the same.

    The article's markdown representation is:
    ```md
    {markdown}
    ```
    """
        )
        .format(markdown=markdown)
        .strip()
    )
    chat_completion = oai.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}], model='gpt-4-1106-preview'
    )
    result = chat_completion.choices[0].message.content
    chunk_size = 4096
    chunks = [
        '\n'.join(filter(bool, result[i : i + chunk_size].splitlines())) for i in range(0, len(result), chunk_size)
    ]
    for i, chunk in enumerate(chunks[:-1]):
        chunk_lines = chunk.splitlines()
        last_chunk_line = chunk_lines[-1]
        chunks[i] = '\n'.join(chunk_lines[:-1])
        chunks[i + 1] = last_chunk_line + chunks[i + 1]
    chunk = '\n'.join(filter(bool, result[:4096].splitlines()))
    audio = oai.audio.speech.create(input=chunk, model='tts-1', voice='alloy', response_format='mp3')
    (Path(__file__).parent / 'we_are_drowning_in_googles_magnanimity.mp3').write_bytes(audio.content)
