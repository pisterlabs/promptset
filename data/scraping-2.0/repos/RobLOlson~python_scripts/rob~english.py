import datetime
import re
import shelve
from pathlib import Path

import appdirs
import toml
import typer
from openai import OpenAI

from . import tomlshelve

app = typer.Typer()

GPT_CLIENT = OpenAI()  # API key must be in environment as "OPENAI_API_KEY"

_THIS_FILE = Path(__file__)

_BOOKS_FILE = (
    Path(appdirs.user_data_dir()) / "robolson" / "english" / "data" / "books.toml"
)
_PROGRESS_FILE = (
    Path(appdirs.user_data_dir()) / "robolson" / "english" / "data" / "progress.toml"
)

_BOOKS_FILE.parent.mkdir(parents=True, exist_ok=True)
_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

_BOOKS_FILE.touch()
_PROGRESS_FILE.touch()

_LATEX_FILE = Path(_THIS_FILE.parent) / "config" / "english" / "latex_templates.toml"
_LATEX_FILE.parent.mkdir(exist_ok=True)
_LATEX_FILE.touch()
LATEX_TEMPLATES = toml.loads(open(_LATEX_FILE.absolute(), "r").read())

_LATEX_PAGE_HEADER = LATEX_TEMPLATES["page_header"]
_LATEX_DOC_HEADER = LATEX_TEMPLATES["doc_header"]

_CONFIG_FILE = Path(_THIS_FILE.parent) / "config" / "english" / "config.toml"
_CONFIG_FILE.parent.mkdir(exist_ok=True)
_CONFIG_FILE.touch()
CONFIGURATION = toml.loads(open(_CONFIG_FILE.absolute(), "r").read())

_WEEKDAYS = CONFIGURATION["constants"]["weekdays"]
_MONTHS = CONFIGURATION["constants"]["months"]
_SYSTEM_INSTRUCTION = CONFIGURATION["constants"]["instruction"]


@app.command("ingest")
def ingest_text_file(target: str, chars_per_page: int = 5_000, debug: bool = False):
    """Import a (properly formatted) book.txt

    line 1: Book Title
    line 2: Book Author
    CHAPTER 1 Optional Chapter Title
    SUBSECTION IN ALL CAPS
    One line of text per paragraph."""

    target = Path(target).name

    fp = open(target, "r", encoding="utf-8")
    lines = fp.readlines()

    title = lines[0].strip()
    author = lines[1].strip()

    text = "\n".join(lines[2:])

    chapter_pattern = re.compile(r"( *CHAPTER (\d+) ?(.*))")
    section_pattern = re.compile(r"\n[^a-z\n\.\?\]\)]+\??\n")

    book = []
    for chapter in chapter_pattern.findall(text):
        chapter_number = int(chapter[1])
        chapter_name = chapter[2]
        full_chapter = f"Chapter {chapter_number}{f': {chapter_name.title()}' if chapter_name else ''}"
        split = text.split(str(chapter[0]))  # chapter[0] is the full match
        chapter_text = split[1]
        chapter_text = chapter_pattern.split(chapter_text)[0]
        subsections = []

        section_titles = section_pattern.findall(chapter_text)
        section_titles = [e.strip() for e in section_titles]
        section_titles.insert(0, full_chapter)
        section_texts = section_pattern.sub(r"SPLIT_CHAPTER_HERE", chapter_text).split(
            "SPLIT_CHAPTER_HERE"
        )

        for i, section_title in enumerate(section_titles):
            subsections.append(
                {"title": section_title, "text": section_texts[i].rstrip()}
            )

        for section in subsections:
            section_length = len(section["text"])
            page_count = section_length // chars_per_page + 1
            if page_count > 1:
                paragraphs = section["text"].split("\n")
                paragraphs = [p for p in paragraphs if p]
                paginated = []
                page_counter = 0
                while paragraphs:
                    new_page = ""
                    page_counter += 1
                    while len(new_page) < section_length / page_count and paragraphs:
                        new_page += f"\n\n \\indent {paragraphs[0]}"
                        del paragraphs[0]
                    doc = {
                        "title": section["title"] + f" (Part {page_counter})",
                        "text": new_page,
                    }

                    paginated.append(doc)

                for part in paginated:
                    book.append(part)
            else:
                book.append(section)

    if not debug:
        with tomlshelve.open(str(_BOOKS_FILE)) as db:
            db[target] = {"book": book, "author": author, "title": title}

    # with tomlshelve.open(str(_BOOKS_FILE) + ".toml") as db:
    #     breakpoint()
    #     db[target] = {"book": book, "author": author, "title": title}


@app.command("set_progress")
def set_progress(target: str, progress: int = 0):
    with tomlshelve.open(str(_PROGRESS_FILE)) as db:
        db[target] = progress


@app.command("generate")
def generate_pages(target: str, n: int = 7, debug: bool = True):
    target = Path(target).name

    with tomlshelve.open(str(_PROGRESS_FILE)) as db:
        if target in db.keys():
            PROGRESS_INDEX = db[target]
        else:
            db[target] = 0
            PROGRESS_INDEX = 0

    mytext = _LATEX_DOC_HEADER
    start = datetime.datetime.today()
    days = [start + datetime.timedelta(days=i) for i in range(30)]
    dates = [
        f"{_WEEKDAYS[day.weekday()]} {_MONTHS[day.month]} {day.day}, {day.year}"
        for day in days
    ]

    # with shelve.open(_BOOKS_FILE.name) as db:
    with tomlshelve.open(str(_BOOKS_FILE)) as db:
        if target not in db.keys():
            print(f"{target} not ingested.")
            return

        book = db[target]["book"]
        title = db[target]["title"]
        author = db[target]["author"]
        book_size = len(book)
        for index in range(n):
            if not debug:
                response = GPT_CLIENT.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": _SYSTEM_INSTRUCTION},
                        {
                            "role": "user",
                            "content": book[(PROGRESS_INDEX + index) % book_size][
                                "text"
                            ],
                        },
                    ],
                )

                questions = str(response.choices[0].message.content)
                lines = questions.split("\n")

                questions = [line for line in lines if len(line) > 0]
                questions = [f"{i+1}.) {text}" for i, text in enumerate(questions)]
                questions = "\n\n".join(questions)
            else:
                questions = "QUESTIONS GO HERE"
            dated_header = re.sub("DATEGOESHERE", dates[index], _LATEX_PAGE_HEADER)
            titled_header = re.sub("TITLEGOESHERE", title, dated_header)
            authored_header = re.sub("AUTHORGOESHERE", author, titled_header)
            mytext += f"{authored_header}\n\\section*{{{book[(PROGRESS_INDEX + index) % book_size]['title']}}}\n\n{book[(PROGRESS_INDEX + index) % book_size]['text']}\n\n{questions}\\newpage"

        mytext = re.sub(pattern=r"&", repl=r"\\&", string=mytext)
        print(mytext)
        fp = open(
            f"English Reading {_MONTHS[datetime.datetime.now().month]} {datetime.datetime.now().day} {datetime.datetime.now().year}.tex",
            "w",
            encoding="utf-8",
        )
        mytext += r"\end{document}"

        fp.write(mytext)

    if not debug:
        with tomlshelve.open(str(_PROGRESS_FILE)) as db:
            db[target] = PROGRESS_INDEX + n


if __name__ == "__main__":
    app()
    # ingest_text_file(debug=False)
    # generate_pages(7, debug=False)
    # db = shelve.open(_BOOKS_FILE.name)
