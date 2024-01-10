from collections.abc import Generator
from pathlib import Path

from langchain import hub
from langchain.prompts import PromptTemplate
from rich.color import Color
from rich.style import Style
from rich.text import Text

from aloud.console import console
from aloud.openai import oai

from . import to_html, to_markdown

# Prompt: TypeAlias = str


# TO_SPEAKABLE: Prompt = """
# You are given a markdown representation of an article from the internet.
#
# Convert the syntax of the markdown into text that can be read out, and keep any real text that is not markdown syntax as-is.
# The general principle is that, as you know, saying "hashtag hashtag <something>" does not make sense to humans, so you should convert that to something like "Moving on to the next part: <something>.", "Next: <something>", "Next Part: <something>", "Next Section: <something>", etc. You can make up new ways to say this. Be creative, mix it up.
# Similarly, saying "Open square brackets, Press here, close square brackets, open parenthesis, https://www.google.com, close parenthesis" does not make sense to humans, so you should convert that to "There's a link to Google here.".
# If a title is followed immediately by a subtitle, say "Moving on to the next part: <title>. Subtitle: <subtitle>".
# Be mindful of how many levels of markdown headers the article contains; Choose the most appropriate way to announce the header, based on its depth. Different depths need to be announced differently.
# Sentences that are completely in emphasis should be announced as "Now, this is important: <sentence>", or "This is important: <sentence>", or "Note, the following is emphasized: <sentence>", or "The following is emphasized: <sentence>", etc (You can make up new ways to say this. Be creative, mix it up).
# Sentences with a word or a short phrase in emphasis should be followed by "the '<emphasized-words-with-hyphens>' part was emphasized".
# If you encounter a table, replace it with "There's a table here that communicates the following information: <information>", where <information> is a short textual summary of the data in the table, and any conclusions that the reader is supposed to draw from it.
# If you encounter code, replace it with "There's a code block here that <description>", where <description> is a short textual description of what the code accepts as input, what it outputs and what takeaway the reader was supposed to get from it. If there are a few code blocks somewhat consecutively, announce the subsequent ones with "Another code block describes <description>", "An additional code block shows <description>", etc.
# Generalize this to the entirety of the markdown syntax.
# The transitions should be smooth and natural.
# Where the text is plain, without any markdown syntax or formatting, keep it exactly the same. Do not summarize.
#
# The article's markdown representation is:
# ```md
# {markdown}
# ```
# """.strip()
# If you encounter a list, append the sentence just before it with a very short saying, communicating that a list is coming up. The saying should meld very organically with the sentence.
TO_SPEAKABLE: PromptTemplate = hub.pull('pecan-ai/aloud-to-speakable')


def to_speakable(thing: str | Path, output_dir: str | Path) -> Generator[str, None, None]:
    output_dir = Path(output_dir)
    if Path(thing).is_file():
        thing = Path(thing).read_text()
    html = to_html(thing, remove_head=True, output_dir=output_dir)
    markdown = to_markdown(html, output_dir=output_dir)
    model = 'gpt-4-1106-preview'
    to_speakable_with_markdown = TO_SPEAKABLE.format(markdown=markdown)
    # prompt = prompts.ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful AI assistant."),
    #         ("human", "{markdown}")
    #     ]
    # )
    # llm = chat_models.ChatOpenAI(model_name=model, temperature=0, streaming=True)
    # chain = prompt | llm | output_parser.StrOutputParser()
    #
    # eval_config = smith.RunEvalConfig(
    #     evaluators=[
    #         "cot_qa",
    #         smith.RunEvalConfig.LabeledCriteria("conciseness"),
    #         smith.RunEvalConfig.LabeledCriteria("relevance"),
    #         smith.RunEvalConfig.LabeledCriteria("coherence")
    #     ],
    #     custom_evaluators=[],
    #     eval_llm=chat_models.ChatOpenAI(model_name="gpt-4", temperature=0)
    # )
    #
    # client = langsmith.Client()
    # chain_results = client.run_on_dataset(
    #     dataset_name="aloud-markdown-to-speakable",
    #     llm_or_chain_factory=chain,
    #     evaluation=eval_config,
    #     project_name="aloud",
    #     concurrency_level=5,
    #     verbose=True,
    # )
    speakable = '\n'
    with console.status(f'Converting markdown to speakable with {model}...') as live:
        start_color = (125, 125, 125)
        end_color = (255, 255, 255)
        for stream_chunk in oai.chat.completions.create(
            messages=[{'role': 'user', 'content': to_speakable_with_markdown}],
            model=model,
            temperature=0,
            stream=True,
        ):
            delta = stream_chunk.choices[0].delta.content or ''
            yield delta
            speakable += delta
            speakable_lines = speakable.splitlines()
            display_speakable = Text()
            num_lines = console.height - 5
            for i, line in enumerate(speakable_lines[-num_lines:]):
                color_rgb = get_gradient_color(start_color, end_color, num_lines - 1, i)
                color = Color.from_rgb(*color_rgb)
                display_speakable += Text(f'{line}\n', style=Style(color=color))
            live.update(display_speakable)
    console.print(speakable)
    speakable_text_path = output_dir / f'{output_dir.name}.txt'
    speakable_text_path.write_text(speakable)
    console.print('\n[b green]Wrote speakable to', speakable_text_path)


def get_gradient_color(start_color, end_color, num_steps, step):
    r_start, g_start, b_start = start_color
    r_end, g_end, b_end = end_color

    r = r_start + (r_end - r_start) * step / num_steps
    g = g_start + (g_end - g_start) * step / num_steps
    b = b_start + (b_end - b_start) * step / num_steps

    return int(r), int(g), int(b)
