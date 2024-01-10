from openai_cli.completion.prompts.generic import ai_prompt


def test_ai_prompt():
    prompt = ai_prompt(
        objective=["write a bash script named {function_name} that does magic."],
    )

    text = prompt.create("abc")

    assert text == "write a bash script named abc that does magic."
