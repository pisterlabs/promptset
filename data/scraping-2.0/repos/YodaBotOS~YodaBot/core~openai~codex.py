from __future__ import annotations

import asyncio
import typing

import openai

if typing.TYPE_CHECKING:
    from . import OpenAI

SUPPORTED_LANGUAGES = [
    "Bash",
    "C",
    "C#",
    "C++",
    "CSS",
    "Dart",
    "Go",
    "HTML",
    "Java",
    "JavaScript",
    "Kotlin",
    "PHP",
    "Perl",
    "Python",
    "R",
    "Ruby",
    "Rust",
    "SQL",
    "Scala",
    "Swift",
    "TypeScript",
]
SUPPORTED_LANGUAGES_LITERAL = typing.Literal[tuple(SUPPORTED_LANGUAGES)]  # type: ignore


class Codex:
    COMMENTS = {
        "Bash": "#",
        "C#": "//",
        "Go": "//",
        "Java": "//",
        "JavaScript": "//",
        "Python": "#",
        "Ruby": "#",
        "Rust": "//",
        "SQL": "--",
        "TypeScript": "//",
    }

    FILE = {
        "Bash": "bash",
        "C#": "cs",
        "Go": "go",
        "Java": "java",
        "JavaScript": "js",
        "Python": "py",
        "Ruby": "rb",
        "Rust": "rs",
        "SQL": "sql",
        "TypeScript": "ts",
    }

    LOWERED_CHARS = {
        "bash": "Bash",
        "c#": "C#",
        "go": "Go",
        "java": "Java",
        "javascript": "JavaScript",
        "python": "Python",
        "ruby": "Ruby",
        "rust": "Rust",
        "sql": "SQL",
        "typescript": "TypeScript",
    }

    MODEL = "gpt-4"

    COMPLETION_KWARGS = {
        "model": MODEL,
        # "temperature": 0,
        # "max_tokens": 512,
        # "top_p": 1,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.3,
    }

    EXPLAIN_KWARGS = {
        "model": MODEL,
        # "temperature": 0,
        # "max_tokens": 100,
        # "top_p": 1,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
        # "stop": ['"""'],
    }

    def __init__(self, openai: OpenAI):
        self.client = openai.client

    @staticmethod
    def _do_comments(language: SUPPORTED_LANGUAGES_LITERAL, text: str) -> str:
        s = ""

        for line in text.splitlines():
            s += f"\n{Codex.COMMENTS[language]} {line}"

        return s.strip()

    @staticmethod
    def _remove_comments(language: SUPPORTED_LANGUAGES_LITERAL, text: str):
        s = ""

        for txt in text.splitlines():
            if txt.startswith(Codex.COMMENTS[language]):
                s += f"\n{txt[len(Codex.COMMENTS[language]):].strip()}"

        return s.strip()

    async def completion(
        self, language: SUPPORTED_LANGUAGES_LITERAL, prompt: str, *, user: int, n: int = 1
    ) -> list[str]:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} is not supported")

        response = await self.client.chat.completions.create(
            **Codex.COMPLETION_KWARGS,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that can help on generating/making code in {language}. You only return the code, nothing else (no explanations, no comments, etc).",
                },
                {"role": "user", "content": prompt},
            ],
            n=n,
            user=str(user),
        )

        choices = []

        for choice in response["choices"]:
            text = choice["message"]["content"].strip()
            choices.append(text)

        return list(set(choices))  # remove duplicates (lazy to do it the hard way)

    async def explain(self, language: SUPPORTED_LANGUAGES_LITERAL, code: str, *, user: int) -> str:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} is not supported")

        response = await self.client.chat.completions.create(
            **Codex.EXPLAIN_KWARGS,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that can help on explaining what a code does in {language}. You only return the explanation, nothing else (no code, etc). Make it a numbered list but make it in order from top to bottom. If any additional notes/explanations are required, put it after the list.",
                },
                {"role": "user", "content": code},
            ],
            user=str(user),
        )

        choice = response["choices"][0]

        text = choice["message"]["content"].strip()

        return text
