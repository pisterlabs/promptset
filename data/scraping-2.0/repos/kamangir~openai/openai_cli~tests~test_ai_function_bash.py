import os
from openai_cli.completion.functions.bash import ai_function_bash
from openai_cli.completion.prompts.bash import bash_prompt
from abcli.modules.objects import select


def test_ai_function():
    select("openai-completion-function-2d-v3")

    prompt = bash_prompt("ingest vancouver.")

    func = ai_function_bash("vancouver_watching")

    assert func.generate(
        prompt.create(
            function_name=func.function_name,
            function_short_name="vanwatch",
        )
    )[0]
