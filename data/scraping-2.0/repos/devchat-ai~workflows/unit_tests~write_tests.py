from functools import partial
from typing import List, Optional

import tiktoken
from chat.ask_codebase.tools.retrieve_file_content import retrieve_file_content
from model import FuncToTest, TokenBudgetExceededException
from openai_util import create_chat_completion_chunks
from prompts import WRITE_TESTS_PROMPT

MODEL = "gpt-4-1106-preview"
TOKEN_BUDGET = int(128000 * 0.9)


def _mk_write_tests_msg(
    root_path: str,
    func_to_test: FuncToTest,
    test_cases: List[str],
    chat_language: str,
    reference_files: Optional[List[str]] = None,
) -> Optional[str]:
    encoding: tiktoken.Encoding = tiktoken.encoding_for_model(MODEL)

    test_cases_str = ""
    for i, test_case in enumerate(test_cases, 1):
        test_cases_str += f"{i}. {test_case}\n"

    reference_content = "\nContent of reference test code:\n\n"
    if reference_files:
        for i, fp in enumerate(reference_files, 1):
            reference_test_content = retrieve_file_content(fp, root_path)
            reference_content += f"{i}. {fp}\n\n"
            reference_content += f"```{reference_test_content}```\n\n"
    else:
        reference_content += "No reference test cases provided.\n\n"

    func_content = f"\nfunction code\n```\n{func_to_test.func_content}\n```\n"
    class_content = ""
    if func_to_test.container_content is not None:
        class_content = f"\nclass code\n```\n{func_to_test.container_content}\n```\n"

    # Prepare a list of user messages to fit the token budget
    # by adjusting the relevant content and reference content
    content_fmt = partial(
        WRITE_TESTS_PROMPT.format,
        function_name=func_to_test.func_name,
        file_path=func_to_test.file_path,
        test_cases_str=test_cases_str,
        chat_language=chat_language,
    )
    # 1. func content & class content & reference file content
    msg_1 = content_fmt(
        relevant_content="\n".join([func_content, class_content]),
        reference_content=reference_content,
    )
    # 2. func content & class content
    msg_2 = content_fmt(
        relevant_content="\n".join([func_content, class_content]),
        reference_content="",
    )
    # 3. func content only
    msg_3 = content_fmt(
        relevant_content=func_content,
        reference_content="",
    )

    prioritized_msgs = [msg_1, msg_2, msg_3]

    for msg in prioritized_msgs:
        tokens = len(encoding.encode(msg))
        if tokens <= TOKEN_BUDGET:
            return msg

    # 3. even func content exceeds the token budget
    raise TokenBudgetExceededException(
        f"Token budget exceeded while writing test cases for <{func_to_test}>. "
        f"({tokens}/{TOKEN_BUDGET})"
    )


def write_and_print_tests(
    root_path: str,
    func_to_test: FuncToTest,
    test_cases: List[str],
    reference_files: Optional[List[str]] = None,
    chat_language: str = "English",
) -> None:
    user_msg = _mk_write_tests_msg(
        root_path=root_path,
        func_to_test=func_to_test,
        test_cases=test_cases,
        reference_files=reference_files,
        chat_language=chat_language,
    )

    chunks = create_chat_completion_chunks(
        model=MODEL,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.1,
    )

    for chunk in chunks:
        if chunk.choices[0].finish_reason == "stop":
            break
        print(chunk.choices[0].delta.content, flush=True, end="")
