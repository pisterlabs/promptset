from typing import Callable

from langchain.chains import LLMChain

from credit_hacking.models import EvalPair
from credit_hacking.prompts import SystemPromptType
from credit_hacking.utils import run_async_with_progress_bar, run_with_progress_bar


async def run_prompts(
    llm: LLMChain, prompts: list[str], system_prompt_type: SystemPromptType, run_async: bool = True
) -> list[EvalPair]:
    """Run the given prompts through the LLM and return the model's answers."""

    if run_async:
        answers: list[str] = await run_async_with_progress_bar(
            [llm.arun(question) for question in prompts], f"Running {system_prompt_type.value} prompts..."
        )
    else:

        def run(question: str) -> Callable[[], str]:
            return lambda: llm.run(question)

        answers = run_with_progress_bar(
            [run(question) for question in prompts], f"Running {system_prompt_type.value} prompts..."
        )

    output: list[EvalPair] = []
    for question, answer in zip(prompts, answers):
        if answer is None:
            raise RuntimeError(f"Task {question} failed to complete")
        output.append(EvalPair(prompt=question, output=answer))

    return output
