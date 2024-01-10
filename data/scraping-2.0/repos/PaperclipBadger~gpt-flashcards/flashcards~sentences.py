import asyncio
import dataclasses
import functools
import itertools
import logging
import pathlib
import re
import sqlite3

import openai

from flashcards.sanitize import strip_html, strip_ruby, strip_parenthetical


logger = logging.getLogger(__name__)


DATABASE = pathlib.Path("sentences.db")
cloze_re = re.compile(r"\{([^\}]*)\}")
comma_re = re.compile(r", |; |、")

# https://platform.openai.com/account/limits
RATE_LIMITS = {
    "gpt-3.5-turbo": 3500,
    "gpt-3.5-turbo-0301": 3500,
    "gpt-3.5-turbo-0613": 3500,
    "gpt-3.5-turbo-1106": 3500,
    "gpt-3.5-turbo-16k": 3500,
    "gpt-3.5-turbo-16k-0613": 3500,
    "gpt-3.5-turbo-instruct": 3000,
    "gpt-3.5-turbo-instruct-0914": 3000,
    "gpt-4": 500,
    "gpt-4-0314": 500,
    "gpt-4-0613": 500,
    "gpt-4-1106-preview": 500,
    "gpt-4-vision-preview": 80,
    "tts-1": 50,
    "tts-1-1106": 50,
    "tts-1-hd": 3,
    "tts-1-hd-1106": 3,
}


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.sem = asyncio.Semaphore(rpm)
        self.rpm = rpm
        self.sleepers = set()
    
    def sleeper_callback(self, task):
        self.sleepers.remove(task)
        self.sem.release()

    async def __aenter__(self):
        await self.sem.acquire()
        task = asyncio.create_task(asyncio.sleep(61))
        self.sleepers.add(task)
        assert len(self.sleepers) <= self.rpm
        task.add_done_callback(self.sleeper_callback)
    
    async def __aexit__(self, exc_type, exc, tb):
        pass


rate_limiters = {}

def get_rate_limiter(model: str):
    try:
        return rate_limiters[model]
    except KeyError:
        rate_limiters[model] = RateLimiter(RATE_LIMITS[model])
        return rate_limiters[model]


def init_cache() -> None:
    DATABASE.touch()
    with sqlite3.connect(DATABASE) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT
            );

            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word INTEGER,
                sentence TEXT,
                FOREIGN KEY(word) REFERENCES words(id)
            );
            """
        )


def encache_sentences(word: str, sentences: list[str]) -> None:
    if not DATABASE.exists():
        init_cache()

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute("SELECT id FROM words WHERE word = ?", (word,))
        if not (row := cursor.fetchone()):
            conn.execute("INSERT INTO words (word) VALUES (?)", (word,))
            cursor = conn.execute("SELECT id FROM words WHERE word = ?", (word,))
            row = cursor.fetchone()
        
        word_id, = row

        for sentence in sentences:
            conn.execute("INSERT INTO sentences (word, sentence) VALUES (?,?)", (word_id, sentence))


def decache_sentences(word: str) -> list[str]:
    if not DATABASE.exists():
        init_cache()
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute(
            """
            SELECT sentences.sentence 
            FROM words 
            LEFT JOIN sentences ON words.id = sentences.word
            WHERE words.word = ?
            """,
            (word,),
        )
        return [row[0] for row in cursor.fetchall() if row[0]]


def uncache_sentences(word: str) -> None:
    if not DATABASE.exists():
        return
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute("SELECT id FROM words WHERE word = ?", (word,))
        if not (row := cursor.fetchone()):
            return
        
        word_id, = row
        conn.execute("DELETE FROM sentences WHERE word = ?", (word_id,))



@dataclasses.dataclass(frozen=True)
class Cloze:
    context: list[str]
    deletions: list[str]

    @classmethod
    def from_str(cls, s: str) -> "Cloze":
        context = []
        deletions = []

        i = 0

        for match in cloze_re.finditer(s):
            context.append(s[i:match.start()])
            deletions.append(match.group(1))
            i = match.end()
        
        context.append(s[i:])

        return cls(context, deletions)

    def fill(self, *values: str) -> str:
        if not values:
            values = ("...",)

        wrapped = ["<b>" + value + "</b>" for value in values]
        parts = itertools.chain.from_iterable(zip(self.context, itertools.cycle(wrapped)))
        return "".join(list(parts)[:2 * len(self.context) - 1])

    @functools.cached_property
    def cloze(self) -> str:
        return self.fill()
    
    @functools.cached_property
    def sentence(self) -> str:
        return self.fill(*self.deletions)


client = None


def init_client():
    global client
    if client is None:
        client = openai.AsyncOpenAI()


async def query_gpt(
    system_prompt: str,
    word: str,
    meaning: str = None,
    model: str = "gtp-4-1106-preview",
) -> list[str]:
    if (cached_sentences := decache_sentences(word)):
        return cached_sentences
    
    init_client()
    
    async with get_rate_limiter(model):
        user_prompt = f"{word} ({meaning})" if meaning else word
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                dict(role="system", content=system_prompt),
                dict(role="user", content=user_prompt),
            ],
            stop="\n\n",
        )

    response = completion.choices[0].message.content.strip()

    if (match := re.match(r"\A\s*(?P<sentence>.+?)\n+(?P<translation>.+?)\s*\Z", response)):
        sentence = match.group("sentence")
        translation = match.group("translation")
    else:
        raise ValueError(f"Could not parse GPT response {response!r}")

    # sometimes GPT forgets to wrap the target word in {}
    # often we can recover by searching for the literal string
    def encloze(sentence: str, words: str) -> str:
        clozed = False

        alternatives = comma_re.split(words)
        for a in alternatives:
            try:
                i = sentence.casefold().index(a.casefold())
            except ValueError as e:
                continue
            else:
                parts = [sentence[:i], "{", a, "}", sentence[i + len(a):]]
                sentence = "".join(parts)
                clozed = True

        if not clozed:
            raise ValueError(f"could not cloze any of {alternatives!r} in {sentence!r}")
        else:
            return sentence

    if not cloze_re.search(sentence):
        plain_word = strip_parenthetical(strip_html(strip_ruby(word)))
        sentence = encloze(sentence, plain_word)
        
    if not cloze_re.search(translation):
        plain_meaning = strip_parenthetical(strip_html(strip_ruby(meaning)))
        try:
            translation = encloze(translation, plain_meaning)
        except ValueError as e:
            logger.error(f"(non-fatal) {e}")
    
    sentences = [f"{sentence}\n{translation}"]

    encache_sentences(word, sentences)

    return sentences


async def example_sentences(
    prompt: str,
    word: str,
    meaning: str = None,
    model: str = "gpt-4-1106-preview",
) -> list[tuple[Cloze, Cloze]]:
    errors = []
    for _ in range(3):
        try:
            sentences = await query_gpt(prompt, word, meaning)
        except ValueError as e:
            logger.error(str(e))
            errors.append(str(e))
            continue

        examples = []
        for pair in sentences:
            sentence, translation = map(Cloze.from_str, pair.split("\n"))
            examples.append((sentence, translation))

        return examples
    else:
        raise ValueError(f"repeatedly queried GPT for {word!r} but got no good responses:" + ", ".join(errors))


count = iter(itertools.count())


async def tts(sentence: str, path: pathlib.Path) -> None:
    if path.exists():
        return

    # drop any html formatting from sentence
    sentence = strip_html(strip_ruby(sentence))

    if not sentence:
        raise Exception("lolwut")

    init_client()

    voice = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"][next(count) % 6]

    async with get_rate_limiter("tts-1"):
        response = await client.audio.speech.create(
            # tts-1-hd is twice as expensive for no noticeable improvement (in Polish).
            model="tts-1",
            voice=voice,
            input=sentence,
            response_format="mp3",
        )
    response.stream_to_file(path)