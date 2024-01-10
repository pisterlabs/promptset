import json
from functools import partial
from typing import List

import tiktoken
from model import FuncToTest, TokenBudgetExceededException
from openai_util import create_chat_completion_content
from prompts import PROPOSE_TEST_PROMPT

MODEL = "gpt-3.5-turbo-1106"
# MODEL = "gpt-4-1106-preview"
TOKEN_BUDGET = int(16000 * 0.9)


def _mk_user_msg(
    user_prompt: str,
    func_to_test: FuncToTest,
    chat_language: str,
) -> str:
    """
    Create a user message to be sent to the model within the token budget.
    """
    encoding: tiktoken.Encoding = tiktoken.encoding_for_model(MODEL)

    func_content = f"function code\n```\n{func_to_test.func_content}\n```\n"
    class_content = ""
    if func_to_test.container_content is not None:
        class_content = f"class code\n```\n{func_to_test.container_content}\n```\n"

    # Prepare a list of user messages to fit the token budget
    # by adjusting the relevant content
    relevant_content_fmt = partial(
        PROPOSE_TEST_PROMPT.format,
        user_prompt=user_prompt,
        function_name=func_to_test.func_name,
        file_path=func_to_test.file_path,
        chat_language=chat_language,
    )
    # 1. func content & class content
    msg_1 = relevant_content_fmt(
        relevant_content="\n".join([func_content, class_content]),
    )
    # 2. func content only
    msg_2 = relevant_content_fmt(
        relevant_content=func_content,
    )

    prioritized_msgs = [msg_1, msg_2]

    for msg in prioritized_msgs:
        token_count = len(encoding.encode(msg))
        if token_count <= TOKEN_BUDGET:
            return msg

    # Even func content exceeds the token budget
    raise TokenBudgetExceededException(
        f"Token budget exceeded while proposing test cases for <{func_to_test}>. "
        f"({token_count}/{TOKEN_BUDGET})"
    )


def propose_test(
    user_prompt: str,
    func_to_test: FuncToTest,
    chat_language: str = "English",
) -> List[str]:
    """Propose test cases for a specified function based on a user prompt

    Args:
        user_prompt (str): The prompt or description for which test cases need to be generated.
        function_name (str): The name of the function to generate test cases for.
        file_path (str): The absolute path to the file containing the target function for which
                         test cases will be generated.

    Returns:
        List[str]: A list of test case descriptions.
    """
    user_msg = _mk_user_msg(
        user_prompt=user_prompt,
        func_to_test=func_to_test,
        chat_language=chat_language,
    )

    content = create_chat_completion_content(
        model=MODEL,
        messages=[{"role": "user", "content": user_msg}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    cases = json.loads(content).get("test_cases", [])

    descriptions = []
    for case in cases:
        description = case.get("description", None)
        if description:
            descriptions.append(description)

    return descriptions
