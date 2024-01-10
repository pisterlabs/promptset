from openai_cli.completion.prompts.bash import bash_prompt
from abcli.modules import objects


function_name = "vancouver_watching"
function_short_name = "vanwatch"


def test_pre_process():
    assert bash_prompt(objects.path_of(f"{function_short_name}-description.txt"))


def test_bash_prompt():
    prompt = bash_prompt("ingest vancouver.")
    prompt.create(
        function_name=function_name,
        function_short_name=function_short_name,
    )
