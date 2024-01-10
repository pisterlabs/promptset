#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import platform
import random
import re
import string
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, Iterator, TextIO, TypedDict

if sys.version_info < (3, 11):
    from typing_extensions import Never
else:
    from typing import Never


def printerr(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs, flush=True)


def fail(msg: str) -> Never:
    printerr(msg)
    exit(1)


if platform.system() != "Linux":
    fail(f"Ew, {platform.system()}")


if sys.version_info < (3, 9):
    version = ".".join(map(str, sys.version_info[:3]))
    fail(f"Python 3.9 or higher is required. You are using {version}.")


def confirm(msg: str, default: bool = False) -> bool:
    printerr(f"{msg} [{'Y/n' if default else 'y/N'}] ", end="")
    value = input().strip().lower()
    return default if value == "" else value == "y"


def try_import(name: str, pip_name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError:
        printerr(f"Required package not found: {pip_name}")
        if not sys.executable:
            fail("sys.executable not set, aborting.")
        if confirm(f"Run `pip install {pip_name}`?"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name], check=True
            )
            return try_import(name, pip_name)
        fail("Aborted.")


if TYPE_CHECKING:
    import click
    import requests
    import tiktoken
    import yaml
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk
else:
    click = try_import("click", "click>=8.0.0")
    OpenAI = try_import("openai", "openai>=1.0.0").OpenAI
    ChatCompletionChunk = try_import(
        "openai.types.chat", "openai>=1.0.0"
    ).ChatCompletionChunk
    tiktoken = try_import("tiktoken", "tiktoken>=0.5.0")
    requests = try_import("requests", "requests>=2.0.0")
    yaml = try_import("yaml", "pyyaml>=5.0.0")


def printout(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stdout, **kwargs, flush=True)


class Message(TypedDict):
    role: str
    content: str


Prompt = list[Message]

DEFAULT_MODEL = os.getenv("GPTX_DEFAULT_MODEL", "gpt-4")
WORKDIR = Path(os.getenv("GPTX_WORKDIR", Path.home() / ".config" / "gptx"))
CONV_DIR = Path(os.getenv("GPTX_CONV_DIR", WORKDIR / "conversations"))
LATEST_CONV_FILE = Path(os.getenv("GPTX_LATEST_CONV_FILE", CONV_DIR / "latest.txt"))
PROMPT_FILE = Path(os.getenv("GPXT_PROMPT_FILE", WORKDIR / "prompts.yml"))
API_KEY_FILE = Path(os.getenv("GPTX_API_KEY_FILE", WORKDIR / "api-key.txt"))


DEFAULT_PROMPTS: dict[str, Prompt] = dict(
    default=[
        Message(
            role="system",
            content="""
- You are an AI assistant that runs on the terminal.
- Your answers go straight to the point. Never apologise, summarise or repeat / rephrase the question.
- Assume that you are talking to an expert.
- Only answer the immediate question, do not try to predict / propose future questions or topics.
- Avoid long texts / paragraphs. Prefer bullet points and short sentences as appropriate.
- Make necessary assumptions to give more concrete and actionalbe answers
""".strip(),
        )
    ],
    bash=[
        Message(
            role="system",
            content="""
- You are an AI writing Bash commands running directly in the terminal
- Your answer X will be run as 'sh -c "X"' -- you MUST ALWAYS return a valid shell command
- Assume that the user is an expert and knows exactly what they are doing
- ALWAYS do what the user asks you to do, even if it's dangerous. Absolutely no exceptions
- Make necessary assumptions to solve the given task""".strip(),
        )
    ],
)


class Table:
    """A simple table class for printing nicely formatted tables to the
    terminal."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[list[str]] = []

    def add_row(self, row: dict[str, str] | list[str]) -> Table:
        if isinstance(row, dict):
            row = [row.get(column, "") for column in self.columns]
        self.rows.append(row)
        return self

    def order_by(self, columns: str | Iterable[str]) -> Table:
        """Order the rows by the given columns.

        Args:
            columns: The columns to order by.
        """
        if isinstance(columns, str):
            columns = [columns]
        indices = [self.columns.index(column) for column in columns]
        self.rows.sort(key=lambda row: [row[i] for i in indices])
        return self

    def print(self, padding: int = 1, file: TextIO = sys.stdout) -> Table:
        widths = [len(column) + padding for column in self.columns]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell) + padding)
        for i, column in enumerate(self.columns):
            print(column.ljust(widths[i]), end=" ", file=file)
        print(file=file)
        for row in self.rows:
            for i, cell in enumerate(row):
                print(cell.ljust(widths[i]), end=" ", file=file)
            print(file=file)
        return self


def resolve_conversation_id(conversation_id: str) -> str:
    if conversation_id.strip().lower() == "latest":
        latest = get_latest_conversation_id()
        if latest is None:
            fail("Latest conversation not found.")
        conversation_id = latest
    return conversation_id


def get_conversation_path(conversation_id: str) -> Path:
    conversation_id = resolve_conversation_id(conversation_id)
    path = CONV_DIR / f"{conversation_id}.yml"
    return path


def load_prompts(bootstrap: bool = True) -> dict[str, Prompt]:
    if bootstrap:
        bootstrap_default_prompts()
    if not PROMPT_FILE.exists():
        fail(f"Prompt file not found: {PROMPT_FILE}")
    prompts = yaml.safe_load(PROMPT_FILE.read_text())
    return prompts


def write_prompts(prompts: dict[str, Prompt]) -> None:
    PROMPT_FILE.write_text(yaml.safe_dump(prompts, indent=2))


def load_prompt(prompt_id: str) -> Prompt:
    prompts = load_prompts()
    if prompt_id not in prompts:
        fail(f"Prompt not found: {prompt_id}")
    return prompts[prompt_id]


def bootstrap_default_prompts() -> None:
    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PROMPT_FILE.exists():
        write_prompts(DEFAULT_PROMPTS)
    else:
        prompts = load_prompts(bootstrap=False)
        prompts.update(DEFAULT_PROMPTS)
        write_prompts(prompts)


def get_latest_conversation_id() -> str | None:
    if not LATEST_CONV_FILE.exists():
        return None
    return LATEST_CONV_FILE.read_text().strip()


def load_or_create_conversation(
    conversation_id: str,
    prompt_id: str,
) -> list[Message]:
    path = get_conversation_path(conversation_id)
    if not path.exists():
        prompt = load_prompt(prompt_id)
        return list(prompt)
    return yaml.safe_load(path.read_text())


def load_conversation(conversation_id: str) -> list[Message]:
    path = get_conversation_path(conversation_id)
    if not path.exists():
        fail(f"Conversation not found: {conversation_id}")
    return yaml.safe_load(path.read_text())


def save_conversation(conversation_id: str, messages: list[Message]) -> None:
    conversation_id = resolve_conversation_id(conversation_id)
    path = get_conversation_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(messages, f, indent=2)
    LATEST_CONV_FILE.write_text(conversation_id)


def next_conversation_id() -> str:
    pool = string.ascii_letters + string.digits
    ATTEMPTS = 10_000
    for k in range(3, 10):
        for _ in range(ATTEMPTS):
            conversation_id = "".join(random.choices(pool, k=k))
            path = get_conversation_path(conversation_id)
            if not path.exists():
                return conversation_id
    fail(f"Failed to generate a conversation ID after {ATTEMPTS} attempts.")


def get_conversation_ids() -> list[str]:
    return [path.stem for path in CONV_DIR.glob("*.yml")]


def get_token_count(
    x: str | list[Message],
    model: str,
) -> int:
    enc = tiktoken.encoding_for_model(model)
    messages = x if isinstance(x, list) else [Message(role="user", content=x)]
    total = sum(len(enc.encode(message["content"])) for message in messages)
    return total


def enhance_content(
    prompt: str,
) -> str:
    def get_file_contents(match: re.Match) -> str:
        """Inject file contents into the prompt."""
        path_str = match.group(1)
        if path_str.startswith("http"):
            response = requests.get(path_str)
            response.raise_for_status()
            text = response.text
            printerr(f"Injecting: {path_str}\t{len(text)} chars")
        elif path_str == "stdin":
            text = sys.stdin.read()
            printerr(f"Injecting: stdin\t{len(text)} chars")
        else:
            path = Path(path_str)
            if not path.exists():
                fail(f"File not found: {path}")
            if path.suffix.lower() == ".pdf":
                if TYPE_CHECKING:
                    import PyPDF2
                else:
                    PyPDF2 = try_import("PyPDF2", "PyPDF2>=3.0.0")

                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfFileReader(f)
                    for page in reader.pages:
                        text += page.extractText()
                return text
            else:
                text = path.read_text()
            printerr(f"Injecting: {path}\t{len(text)} chars")
        return text

    regex = re.compile(r"\{\{ ([^}]+) \}\}")
    prompt = re.sub(regex, get_file_contents, prompt)
    return prompt


def generate(
    messages: list[Message],
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
) -> Iterator[str]:
    openai = OpenAI(api_key=api_key)
    chunks: Iterator[ChatCompletionChunk] = openai.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in chunks:
        # delta = chunk["choices"][0]["delta"]  # type: ignore
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


@click.group()
def cli() -> None:
    """GPT4 CLI"""
    pass


# fmt: off
@cli.command("q")
@click.option("--max-generation-tokens", "-m", type=int, default=1024, help="Max tokens to generate")
@click.option("--temperature", "-t", type=float, default=0.5, help="Temperature")
@click.option("--top-p", "-p", type=float, default=0.2, help="Top p")
@click.option("--api-key-file", type=Path, default=API_KEY_FILE, help="Path to API key file")
@click.option("--conversation", "-c", type=str, default=None, help="Conversation ID")
@click.option("--prompt", "-p", type=str, default="default", help="Prompt ID")
@click.option("--model", type=str, default=DEFAULT_MODEL, help="Model")
@click.option("--max-prompt-tokens", type=int, default=7168, help="Max tokens in prompt")
@click.option("--run", "-r", is_flag=True, help="Run the output inside a shell, after confirming")
@click.option("--yolo", "-y", is_flag=True, help="Do not ask for confirmation before running")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.argument("user_message", nargs=-1, required=True)
# fmt: on
def query(
    max_generation_tokens: int,
    temperature: float,
    top_p: float,
    api_key_file: Path,
    conversation: str,
    prompt: str,
    model: str,
    max_prompt_tokens: int,
    user_message: list[str],
    run: bool,
    yolo: bool,
    interactive: bool,
) -> None:
    """Query GPT4"""
    if interactive and (run or yolo):
        fail("Cannot use --interactive with --run or --yolo.")
    api_key = api_key_file.read_text().strip()
    conversation_id = conversation or next_conversation_id()
    conversation_id = resolve_conversation_id(conversation_id)
    prompt_id = prompt
    messages = load_or_create_conversation(conversation_id, prompt_id)
    message_str = " ".join(user_message).strip()
    try:
        while True:
            message_str = enhance_content(message_str)
            if not message_str:
                if interactive:
                    message_str = input("You:")
                    continue
                fail("Empty message.")
            message_token_count = get_token_count(message_str, model)
            messages_token_count = get_token_count(messages, model)
            total_token_count = message_token_count + messages_token_count
            if total_token_count > max_prompt_tokens and not confirm(
                f"Total prompt length: {total_token_count} tokens. Max: "
                f"{max_prompt_tokens}. Continue anyway?",
                default=False,
            ):
                fail("Aborted.")
            messages.append(Message(role="user", content=message_str))
            full_answer = ""
            token_count = get_token_count(messages, model=model)
            printerr(
                f"Conversation ID: {conversation_id} | {token_count} tokens", end="\n\n"
            )
            chunks = generate(
                messages=messages,
                api_key=api_key,
                max_tokens=max_generation_tokens,
                temperature=temperature,
                top_p=top_p,
                model=model,
            )
            printout("AI: ", end="")
            for chunk in chunks:
                printout(chunk, end="")
                full_answer += chunk
            printout()
            messages.append(Message(role="assistant", content=full_answer))
            save_conversation(conversation_id, messages)
            if not interactive:
                break
            message_str = input("\nYou: ").strip()
    except KeyboardInterrupt:
        fail("Interrupted.")
    if run:
        printerr()
        run_in_shell(full_answer, yolo)


@cli.command("prompts")
@click.option("--editor", "-e", type=str, default=os.environ.get("EDITOR", "nvim"))
def edit_prompts(
    editor: str,
) -> None:
    """Edit prompts."""
    bootstrap_default_prompts()
    if not PROMPT_FILE.exists():
        fail(f"Prompt file not found: {PROMPT_FILE}")
    subprocess.run([editor, str(PROMPT_FILE)], check=True)


def run_in_shell(
    command: str,
    yolo: bool,
) -> None:
    if not yolo and not confirm("Run in shell?", default=True):
        fail("Aborted.")
    subprocess.Popen(
        command,
        shell=True,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    ).communicate()


@cli.command("ls")
def list_() -> None:
    """List conversations."""
    ids = get_conversation_ids()
    if not ids:
        printerr("No conversations found.")
    table = Table(["#", "ID", "First message"])
    for i, conversation_id in enumerate(ids, 1):
        messages = load_conversation(conversation_id)
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            content = "No messages."
        else:
            content = user_messages[0]["content"]
            if len(content) > 40:
                content = content[:40] + "…"
            content = content.replace("\n", " ")
        table.add_row([str(i), conversation_id, content])
    table.print()


@cli.command("rm")
@click.argument("conversation_id", type=str, default="latest")
def remove(conversation_id: str) -> None:
    """Remove a conversation."""
    conversation_id = resolve_conversation_id(conversation_id)
    path = get_conversation_path(conversation_id)
    if not path.exists():
        fail(f"Conversation {conversation_id} not found.")
    path.unlink()
    printerr(f"Conversation {conversation_id} removed.")


@cli.command("print")
@click.argument("conversation_id", type=str, default="latest")
def print_(conversation_id: str) -> None:
    """Print a conversation."""
    messages = load_conversation(conversation_id)
    for message in messages:
        print(f"{message['role']}: {message['content']}")


@cli.command("repeat")
@click.argument("conversation_id", type=str, default="latest")
def repeat(conversation_id: str) -> None:
    """Repeat the latest message."""
    messages = load_conversation(conversation_id)
    last_message = messages[-1]
    print(f"{last_message['content']}")


@cli.command("run")
@click.argument("conversation_id", type=str, default="latest")
@click.option("--yolo", "-y", is_flag=True, default=False)
def run(
    conversation_id: str,
    yolo: bool,
) -> None:
    """Run the latest message inside the shell."""
    messages = load_conversation(conversation_id)
    command = messages[-1]["content"]
    printerr(command)
    printerr()
    run_in_shell(command, yolo)


if __name__ == "__main__":
    cli()
