"""runs tests against llm_logic source code"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.llm_logic import (
    transcribe_audio,
    call_completions
)

TRANSCRIPTION_MODEL = "whisper-1"
COMPLETIONS_MODEL = "gpt-3.5-turbo-1106"
with open("tests/transcription_test/transcription.txt", "r", encoding = "utf8") as file:
    TRANSCRIPTION = file.read()

load_dotenv()
client = OpenAI(
	api_key = f"{os.environ.get("OPENAI_API_KEY")}"
)
ISSUE_RESOLVED_PROMPT = os.environ.get("IS_ISSUE_RESOLVED_PROMPT")
OFFERING_PROMPT = os.environ.get("OFFERING_PROMPT")


def test_transcribe_audio() -> None:
    """tests to see if transcription is successful"""
    transcription = transcribe_audio(
        client = client,
        model = TRANSCRIPTION_MODEL,
        audio_file = "tests/audio_test/How to Pronounce Hello.mp4"
    )
    assert transcription is not None


def test_call_completions_issue_resolved_prompt() -> None:
    """tests to see if the prompt evaluates if they issue has been resolved correctly"""
    result = call_completions(
        client = client,
        model = COMPLETIONS_MODEL,
        prompt = ISSUE_RESOLVED_PROMPT,
        temperature = 0,
        input_text = TRANSCRIPTION
    )
    assert int(result) == 1


def test_call_completions_offering_prompt() -> None:
    """tests to see if the prompt evaluates if offering has been given correctly"""
    result = call_completions(
        client = client,
        model = COMPLETIONS_MODEL,
        prompt = OFFERING_PROMPT,
        temperature = 0,
        input_text = TRANSCRIPTION
    )
    assert int(result) == 1
