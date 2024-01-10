import math

import numpy as np
import pytest

from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.structs import LmPrompt

ALL_MODELS = [
    get_open_ai_lm(OpenAiModelNames.text_ada_001),
    get_huggingface_lm("gpt2"),
]


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred(lm):
    out = lm.predict(
        LmPrompt(
            "Here is a story. Once upon a",
            max_tokens=1,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "time"


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred_lp(lm):
    out = lm.predict(
        LmPrompt(
            "Here is a story. Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=False,
        ),
    )
    assert out.completion_text.strip() == "time"
    print(out)
    assert lm.remove_special_chars_from_tokens(out.completion_tokens) == [" time"]
    assert len(out.completion_logprobs) == 1
    assert math.exp(out.completion_logprobs[0]) >= 0.9


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred_cache(lm):
    runtimes = []
    import time

    for _i in range(2):
        start = time.time()
        out = lm.predict(
            LmPrompt(
                "Once upon a",
                max_tokens=1,
                logprobs=1,
                cache=True,
                num_completions=1,
                echo=False,
            ),
        )
        end = time.time()
        assert out.completion_text.strip() == "time"
        runtimes.append(end - start)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_echo(lm):
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=True,
        ),
    )
    print(out.get_full_text())
    assert out.get_full_text().strip() == "Once upon a time"
    assert out.completion_text.strip() == "time"
    assert lm.remove_special_chars_from_tokens(out.prompt_tokens) == [
        "Once",
        " upon",
        " a",
    ]
    assert len(out.prompt_logprobs) == 3
    assert len(out.prompt_logprobs) == 3
    assert len(out.full_logprobs) == 4
    assert lm.remove_special_chars_from_tokens(out.get_full_tokens()) == [
        "Once",
        " upon",
        " a",
        " time",
    ]


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_low_prob_in_weird_sentence(lm):
    weird = lm.predict(
        LmPrompt(
            "The Empire State Building is in New run and is my favorite",
            max_tokens=1,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=True,
        ),
    )
    normal = lm.predict(
        LmPrompt(
            "The Empire State Building is in New York and is my favorite",
            max_tokens=1,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=True,
        ),
    )
    no_space = lm.remove_special_chars_from_tokens(weird.prompt_tokens)
    assert no_space == [
        "The",
        " Empire",
        " State",
        " Building",
        " is",
        " in",
        " New",
        " run",
        " and",
        " is",
        " my",
        " favorite",
    ]
    assert len(weird.prompt_logprobs) == len(weird.prompt_tokens)
    weird_idx = no_space.index(" run")
    assert math.exp(weird.prompt_logprobs[weird_idx]) < 0.001
    assert math.exp(normal.prompt_logprobs[weird_idx]) > 0.5
    assert math.exp(weird.prompt_logprobs[weird_idx]) < math.exp(
        normal.prompt_logprobs[weird_idx],
    )
    assert math.exp(weird.prompt_logprobs[weird_idx - 1]) == pytest.approx(
        math.exp(normal.prompt_logprobs[weird_idx - 1]),
        rel=1e-5,
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_no_gen(lm):
    val = lm.predict(
        LmPrompt(
            "I like pie",
            max_tokens=0,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=True,
        ),
    )
    assert len(val.prompt_tokens) == 3
    assert len(val.prompt_logprobs) == 3
    assert len(val.completion_tokens) == 0
    assert len(val.completion_text) == 0
    assert len(val.completion_logprobs) == 0


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_many_gen(lm):
    val = lm.predict(
        LmPrompt(
            "Write a story about a pirate:",
            max_tokens=5,
            logprobs=1,
            cache=False,
        ),
    )
    assert len(val.completion_tokens) == 5


@pytest.mark.parametrize("lm", ALL_MODELS)
@pytest.mark.skip(
    reason=(
        "OpenAI will insert an <|endoftext|> when doing"
        "unconditional generation and need to look into if also"
        "happens with the chat models and how to handle it"
    ),
)
def test_unconditional_gen(lm):
    # TODO: handle for openai
    val = lm.predict(
        LmPrompt(
            "",
            max_tokens=2,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    assert len(val.prompt_tokens) == 0
    assert len(val.prompt_logprobs) == 0
    assert len(val.completion_tokens) == 2
    assert len(val.completion_text) > 2
    assert len(val.completion_logprobs) == 2


capital_prompt = (
    "The capital of Germany is the city Berlin. "
    "The capital of Spain is the city Madrid. "
    "The capital of UK is the city London. "
    "The capital of France"
)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_no_stopping_in_prompt(lm):
    capital_newlines = (
        "The capitol of Germany\n is the city Berlin.\n"
        "The capital of Spain\n is the city Madrid.\n"
        "The capital of UK\n is the city London.\n"
        "The capital of France\n"
    )
    # By having the last character of the prompt be a stop token
    # we ensure that stopping logic does not include the prompt
    new_line = lm.predict(
        LmPrompt(
            capital_newlines,
            stop=["\n"],
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert len(new_line.completion_tokens) == 4


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_no_stopping_program(lm):
    prompt_text = "# Functions\ndef double(x):\n"
    resp = lm.predict(
        LmPrompt(
            prompt_text,
            max_tokens=50,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert "\ndef" in resp.completion_text
    print(resp.completion_text)
    resp = lm.predict(
        LmPrompt(
            prompt_text,
            stop=["\ndef"],
            max_tokens=50,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert "\ndef" not in resp.completion_text


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_begin_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    print(val_normal.completion_text)
    assert "is the city Paris" in val_normal.completion_text
    assert len(val_normal.completion_tokens) == 4
    assert (
        lm.remove_special_chars_from_tokens(val_normal.completion_tokens)[-1]
        == " Paris"
    )
    # Chopping off first part of subtoken does not return token
    val_no_pa = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=[" Pa"],
        ),
    )
    print(val_no_pa.completion_text)
    assert val_no_pa.completion_text == " is the city"
    assert len(val_no_pa.completion_tokens) == 3
    assert np.allclose(
        val_no_pa.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_middle_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off middle of subtoken returns token but cut
    val_no_ari = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ari"],
        ),
    )
    assert val_no_ari.completion_text == " is the city P"
    assert len(val_no_ari.completion_logprobs) == 4
    assert np.allclose(
        val_no_ari.completion_logprobs,
        val_normal.completion_logprobs,
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ari.completion_tokens)[-1]
        == " Paris"
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_end_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off end of subtoken returns token but cut
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ris"],
        ),
    )
    assert val_no_ris.completion_text == " is the city Pa"
    assert len(val_no_ris.completion_logprobs) == 4
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs,
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1]
        == " Paris"
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_span_subtoks(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off between multiple subtokens
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=10,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ity Paris"],
        ),
    )
    assert val_no_ris.completion_text == " is the c"
    assert len(val_no_ris.completion_logprobs) == 3
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1] == " city"
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_span_subtoks2(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off between multiple subtokens in middle
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=10,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ity Par"],
        ),
    )
    assert val_no_ris.completion_text == " is the c"
    assert len(val_no_ris.completion_logprobs) == 3
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1] == " city"
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_stopping_span_subtoks_multiple(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    for do_reverse in [True, False]:
        stop = ["ity Par", "ty P"]
        if do_reverse:
            stop.reverse()
        val_no_ris = lm.predict(
            LmPrompt(
                capital_prompt,
                max_tokens=10,
                logprobs=1,
                temperature=0,
                cache=False,
                stop=stop,
            ),
        )
        assert val_no_ris.completion_text == " is the c"
        assert len(val_no_ris.completion_logprobs) == 3
        assert np.allclose(
            val_no_ris.completion_logprobs,
            val_normal.completion_logprobs[:-1],
            atol=0.001,
            rtol=0.001,
        )
        assert (
            lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1]
            == " city"
        )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_remove_prompt_from_cache(lm):
    prompt = LmPrompt(
        "Give a random base-64 guid:",
        max_tokens=100,
        temperature=2.0,
        cache=True,
    )
    r1 = lm.predict(prompt)
    r2 = lm.predict(prompt)
    assert r1.completion_text == r2.completion_text
    assert lm.remove_prompt_from_cache(prompt)
    r3 = lm.predict(prompt)
    if r1.completion_text != r3.completion_text:
        return  # Pass because it is different and uncached
    # Sometimes still flacky with just one prompt
    assert lm.remove_prompt_from_cache(prompt)
    r4 = lm.predict(prompt)
    assert (
        r1.completion_text != r3.completion_text
        or r1.completion_text != r4.completion_text
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_none_max_tokens(lm):
    prompt = LmPrompt(
        "Write a long and detailed story about a dog:",
        max_tokens=None,
        temperature=1.0,
        cache=False,
    )
    result = lm.predict(prompt)
    assert len(result.completion_tokens) == lm.default_tokens_generated
