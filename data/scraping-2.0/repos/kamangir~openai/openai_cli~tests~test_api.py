from openai_cli.completion.api import complete_prompt


def test_complete_prompt():
    prompt = "Write a tag line for an cafe in Vancouver"
    max_tokens = 1000
    verbose = True

    result = complete_prompt(prompt, max_tokens, verbose)

    assert isinstance(result, tuple)
    assert len(result) == 3

    assert result[0] is True
    assert isinstance(result[1], str)
    assert isinstance(result[2], dict)
