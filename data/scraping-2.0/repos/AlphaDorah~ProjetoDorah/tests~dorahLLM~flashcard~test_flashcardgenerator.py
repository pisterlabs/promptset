import pytest
from src.dorahLLM.flashcard.flashcard import Flashcard

from src.dorahLLM.flashcard.flashcardgenerator import MaritalkFlashcardGenerator
from langchain.llms.fake import FakeListLLM
from langchain.llms.base import LLM

import pytest


def test_empty_text():
    generator = MaritalkFlashcardGenerator(model=FakeListLLM(responses=[]))
    text = ""
    flashcards = generator._parse_flashcards(text)
    assert flashcards == []


def test_single_flashcard():
    generator = MaritalkFlashcardGenerator(model=FakeListLLM(responses=[]))
    text = "Pergunta: What is the capital of France?\nResposta: Paris"
    flashcards = generator._parse_flashcards(text)
    assert flashcards == [Flashcard("What is the capital of France?", "Paris")]


def test_multiple_flashcards():
    generator = MaritalkFlashcardGenerator(model=FakeListLLM(responses=[]))
    text = "Pergunta: What is the capital of France?\nResposta: Paris\nPergunta: What is the capital of Germany?\nResposta: Berlin"
    flashcards = generator._parse_flashcards(text)
    assert flashcards == [
        Flashcard("What is the capital of France?", "Paris"),
        Flashcard("What is the capital of Germany?", "Berlin"),
    ]


def test_missing_answer():
    generator = MaritalkFlashcardGenerator(model=FakeListLLM(responses=[]))
    text = "Pergunta: What is the capital of France?"
    flashcards = generator._parse_flashcards(text)
    assert flashcards == []


def test_missing_question():
    generator = MaritalkFlashcardGenerator(model=FakeListLLM(responses=[]))
    text = "Resposta: Paris"
    flashcards = generator._parse_flashcards(text)
    assert flashcards == []


class TestGenerate:
    def test_valid_summary(self):
        # Test case 1: Verify that the function returns the expected text when given a valid summary.
        responses = [
            "Pergunta: This is a valid summary?\nResposta: Yes.",
        ]
        instance = MaritalkFlashcardGenerator(model=FakeListLLM(responses=responses))
        summary = "This is a valid summary."
        expected_flashcard = Flashcard(
            question="This is a valid summary?", answer="Yes."
        )
        result = instance.generate(summary)
        assert result == [expected_flashcard]
