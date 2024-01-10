import pytest

from anki_convo.card import TextCard
from anki_convo.chains.llm import LLMChain
from anki_convo.factory import get_card_side, get_prompt
from anki_convo.llm_filter import (
    CardGenerator,
    CardGeneratorConfig,
    ChainCardGenerator,
    LLMFilter,
    LLMFilterConfig,
    parse_llm_filter_name,
    parse_text_card_response,
)
from anki_convo.llms.openai import OpenAI


def test_parse_llm_filter_name_valid():
    filter_name = "llm vocab-to-sentence lang_front=English lang_back=Ukrainian front"
    result = parse_llm_filter_name(filter_name)
    expected = LLMFilterConfig(
        card_generator=CardGeneratorConfig(
            prompt_name="vocab-to-sentence", lang_front="English", lang_back="Ukrainian"
        ),
        card_side="front",
    )

    assert result == expected


def test_parse_llm_filter_name_invalid():
    filter_name = "invalid_filter_name"
    with pytest.raises(ValueError):
        parse_llm_filter_name(filter_name)


@pytest.mark.parametrize(
    ("response", "expected_cards"),
    [
        ("[]", []),
        (
            '[{"front": "Hello, how are you?", "back": "Привіт, як справи?"}]',
            [TextCard(front="Hello, how are you?", back="Привіт, як справи?")],
        ),
        (
            (
                '[{"front": "Hello, how are you?", "back": "Привіт, як справи?"}, '
                '{"front": "Good morning, have a nice day!",'
                ' "back": "Доброго ранку, маєте чудовий день!"}]'
            ),
            [
                TextCard(front="Hello, how are you?", back="Привіт, як справи?"),
                TextCard(
                    front="Good morning, have a nice day!",
                    back="Доброго ранку, маєте чудовий день!",
                ),
            ],
        ),
        # Weirdly formatted JSON
        ("[         ]", []),
        (
            (
                '[          {  \n"front": "Hello, how are you?"  ,'
                '\n\n"back":   "Привіт, як справи?"},   '
                '\n\n\t{"front":    "Good morning, have a nice day!",'
                '\n"back": "Доброго ранку, маєте чудовий день!"}\n\n\n]'
            ),
            [
                TextCard(front="Hello, how are you?", back="Привіт, як справи?"),
                TextCard(
                    front="Good morning, have a nice day!",
                    back="Доброго ранку, маєте чудовий день!",
                ),
            ],
        ),
    ],
)
def test_parse_text_card_response_valid(response, expected_cards):
    actual_cards = parse_text_card_response(response)
    assert actual_cards == expected_cards


@pytest.fixture(params=[0, 1, 3, 5])
def n_cards(request):
    return request.param


@pytest.fixture()
def language_card_generator(n_cards):
    chain = LLMChain(
        llm=OpenAI(model="gpt-3.5-turbo"), prompt=get_prompt("vocab-to-sentence")
    )

    generator = ChainCardGenerator(
        chain=chain,
        chain_input={
            "lang_front": "English",
            "lang_back": "Ukrainian",
            "n_cards": n_cards,
        },
    )
    return generator


@pytest.mark.slow
@pytest.mark.expensive
@pytest.mark.parametrize("field_text", ["friend", "to give", "love"])
def test_language_card_generator(language_card_generator, field_text):
    actual_cards = language_card_generator(field_text)

    # TODO The language card generator is non-deterministic, so we cannot come up with
    #   fixed unit tests for this. One possibility is to judge the quality of the output
    #   through a metric like BLEU score, or let a powerful LLM like GPT-4 rate the
    #   output based on some rules.
    #   We can also try PromptWatch.

    assert len(actual_cards) == language_card_generator.n_cards
    for card in actual_cards:
        assert isinstance(card, TextCard)


class CountingCardGenerator(CardGenerator):
    """Card generator that returns a fixed number of cards with the same front and back.

    It keeps track of how often it was called in order to test that lru_cache works.
    """

    def __init__(self, n_cards: int = 1):
        self.n_cards = n_cards
        self.n_times_called = 0

    def __call__(self, field_text):
        self.n_times_called += 1
        front = f"Front of card for '{field_text}' after {self.n_times_called} call(s)"
        back = f"Back of card for '{field_text}' after {self.n_times_called} call(s)"

        return [
            TextCard(front=f"{front} (card {i})", back=f"{back} (card {i})")
            for i in range(self.n_cards)
        ]


@pytest.fixture()
def counting_card_generator(n_cards):
    card_generator = CountingCardGenerator(n_cards=n_cards)

    return card_generator


@pytest.fixture(params=["front", "back"])
def card_side(request):
    return get_card_side(request.param)


@pytest.fixture()
def llm_filt(counting_card_generator, card_side):
    filt = LLMFilter(card_generator=counting_card_generator, card_side=card_side)
    return filt


# @pytest.mark.parametrize(())
# def test_llm_filter(llm_filt: LLMFilter, field_text, field_name, expected):
#     if llm_filt.card_generator.n_cards == 0:
#         return field_text

#     result = llm_filt(field_text=field_text, field_name="Front")

#     if counting_card_generator.n_cards == 0:
#         pass

#     # Create an LLMFilter instance
#     filter = LLMFilter(mock_card_generator, CardSide.FRONT)

#     # Set up the mock to return a TextCard with the expected front and back
#     mock_card = MagicMock()
#     mock_card.front = "Hello, how are you?"
#     mock_card.back = "Привіт, як справи?"
#     mock_card_generator.return_value = [mock_card]

#     # Call the filter with some field text and name
#     field_text = "Hello"
#     field_name = "Front"
#     result = filter(field_text, field_name)

#     # Check that the mock was called with the expected field text
#     mock_card_generator.assert_called_once_with(field_text=field_text)

#     # Check that the result is the expected card side
#     assert result == mock_card.front
