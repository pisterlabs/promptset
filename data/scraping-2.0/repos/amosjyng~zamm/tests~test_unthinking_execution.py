import langchain_visualizer  # isort:skip  # noqa: F401
import pytest
import vcr_langchain as vcr
from langchain.llms import OpenAI
from tiktoken_ext.openai_public import p50k_base

from zamm.agents.employee import ZammEmployee


@pytest.mark.skip("impossible to get GPT3 to follow directions here")
async def test_execute_goodbye_task():
    p50k_base()  # run this before cassette to download tiktoken blob first

    with open("zamm/resources/tutorials/hello.md") as tutorial:
        with vcr.use_cassette("tests/resources/goodbye-unthinking.yaml"):
            llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=-1)
            employee = ZammEmployee(
                llm=llm, terminal_safe_mode=True, think_before_acting=False
            )
            results = employee(
                {
                    "task": (
                        'Write a script goodbye.sh that prints out "Goodbye world". '
                        "Only edit that file a single time. Execute it right after "
                        "file creation."
                    ),
                    "documentation": tutorial.read(),
                }
            )
            assert results is not None


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_execute_goodbye_task)
