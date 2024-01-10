import os

import openai
import pytest


def pytest_sessionstart() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")


@pytest.fixture
def training_data() -> list:
    """Mock training data to use as few tokens as possible."""
    return [
        {
            "prompt": "A",
            "completion": "B",
        },
    ]


@pytest.fixture
def file_id() -> str:
    """File ID for an uploaded file in the OpenAI API."""
    return "file-TnIO5MBmKZmzOtpM4ISCkEwx"


@pytest.fixture
def fine_tune_id() -> str:
    """Fine-tune ID for the IMDB text classification model trained in `examples/text_classification_imdb.ipynb`.
    """
    return "ft-nbCAuynTlICBi7HfJ82XFB9F"


@pytest.fixture
def fine_tuned_model() -> str:
    """Fine-tuned model for the IMDB text classification model trained in `examples/text_classification_imdb.ipynb`.
    """
    return "curie:ft-personal-2023-04-19-12-40-56"


@pytest.fixture
def prompt() -> str:
    """Prompt generated for a random entry in the IMDB test set."""
    return (
        "Categorize the following text from an IMDB review in the following categories"
        " based on its sentiment: 'pos', or 'neg'.\n\nReview: I love sci-fi and am"
        " willing to put up with a lot. Sci-fi movies/TV are usually underfunded,"
        " under-appreciated and misunderstood. I tried to like this, I really did, but"
        " it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly"
        " prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match"
        " the background, and painfully one-dimensional characters cannot be overcome"
        " with a 'sci-fi' setting. (I'm sure there are those of you out there who think"
        " Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While"
        " US viewers might like emotion and character development, sci-fi is a genre"
        " that does not take itself seriously (cf. Star Trek). It may treat important"
        " issues, yet not as a serious philosophy. It's really difficult to care about"
        " the characters here as they are not simply foolish, just missing a spark of"
        " life. Their actions and reactions are wooden and predictable, often painful"
        " to watch. The makers of Earth KNOW it's rubbish as they have to always say"
        " 'Gene Roddenberry's Earth...' otherwise people would not continue watching."
        " Roddenberry's ashes must be turning in their orbit as this dull, cheap,"
        " poorly edited (watching it without advert breaks really brings this home)"
        " trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main"
        " character. And then bring him back as another actor. Jeeez! Dallas all over"
        " again.\n\nLabel: "
    )
