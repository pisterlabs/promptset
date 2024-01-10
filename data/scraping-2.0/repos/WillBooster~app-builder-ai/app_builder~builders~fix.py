import logging
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.vectorstores import FAISS
from parsers.strict_pydantic_output_parser import StrictPydanticOutputParser
from schema import (
    PublicInterfaceDocument,
    SourceCodeFix,
    SourceCodeFixOption,
    SourceCodeFixOptionSet,
)
from util import (
    RAW_ALL_TEST_ID,
    TEST_LOG_FILE_NAME,
    execute_model,
    get_doc_file_path,
    get_prompt_file_path,
    get_src_file_path,
)


def fix_test_errors(
    model: ChatOpenAI,
    specifications_text: str,
    public_interface_document: PublicInterfaceDocument,
    source_code_vector_db: FAISS,
    test_failures: dict[str, str],
) -> list[str]:
    logging.info("Fixing test errors.")
    fixed_file_names = []

    for test_id, error_message in test_failures.items():
        if test_id == RAW_ALL_TEST_ID:
            continue
        test_file_name = f"{test_id.split('.')[0]}.py"
        option_collection = suggest_source_code_fixes(
            model,
            source_code_vector_db,
            public_interface_document,
            test_file_name,
            error_message,
            test_failures[RAW_ALL_TEST_ID],
        )

        print(f"We are now trying to fix for acceptance test {test_id}.")
        print(f"Error message:\n{error_message}\n")
        for i, option in enumerate(option_collection.options):
            print(f"Fix {i + 1} - {option.file_name}:")
            print(option.observation)
            print(option.how_to_fix)
            print()
        print(f"Fix {len(option_collection.options) + 1} - Specify a fix manually.")

        selected_fix_idx = input(
            "Which fix do you want to apply? (Enter a number): "
        ).strip()
        while (
            not selected_fix_idx.isdigit()
            or int(selected_fix_idx) > len(option_collection.options) + 1
        ):
            selected_fix_idx = input(
                f"Invalid input. Please enter a number between 1 and "
                f"{len(option_collection.options) + 1}: "
            )

        if int(selected_fix_idx) == len(option_collection.options) + 1:
            print("Please specify a fix manually.")
            print("File name:")
            file_name = input()
            print("Observation:")
            observation = input()
            print("How to fix:")
            how_to_fix = input()
            option = SourceCodeFixOption(
                file_name=file_name,
                observation=observation,
                how_to_fix=how_to_fix,
            )
        else:
            option = option_collection.options[int(selected_fix_idx) - 1]

        source_code_fix = gen_source_code_fix_from_plan(
            model,
            option,
            option.file_name,
            test_file_name,
            error_message,
            specifications_text,
            public_interface_document,
        )

        apply_source_code_fix(source_code_fix)
        fixed_file_names.append(source_code_fix.file_name)

    return fixed_file_names


def suggest_source_code_fixes(
    model: ChatOpenAI,
    source_code_vector_db: FAISS,
    public_interface_document: PublicInterfaceDocument,
    test_file_name: str,
    error_message: str,
    raw_all_test_log: str,
) -> SourceCodeFixOptionSet:
    logging.info(f"Generating source code fix for {test_file_name}.")

    source_code_docs = source_code_vector_db.similarity_search(error_message, k=3)
    source_code_dataset = "\n".join(
        [
            f"{os.path.basename(doc.metadata['source'])}\n"
            f"```\n{doc.page_content}\n```\n"
            for doc in source_code_docs
        ]
    )

    test_code = open(get_src_file_path(test_file_name)).read()

    output_parser = StrictPydanticOutputParser(pydantic_object=SourceCodeFixOptionSet)
    prompt = load_prompt(get_prompt_file_path("suggest_test_fixes.yaml")).format(
        error_message=raw_all_test_log,
        test_file=test_file_name,
        test_code=test_code,
        public_interface_document=public_interface_document.json(),
        source_code_dataset=source_code_dataset,
        format_instructions=output_parser.get_format_instructions(),
    )
    output = execute_model(model, prompt)
    source_code_fix = output_parser.parse(output)
    return source_code_fix


def gen_source_code_fix_from_plan(
    model: ChatOpenAI,
    source_code_fix_option: SourceCodeFixOption,
    fixed_file_name: str,
    test_file_name: str,
    error_message: str,
    specifications_text: str,
    public_interface_document: PublicInterfaceDocument,
) -> SourceCodeFix:
    logging.info(f"Generating source code fix for {fixed_file_name}.")

    fixed_code = open(get_src_file_path(fixed_file_name)).read()
    test_code = open(get_src_file_path(test_file_name)).read()

    plan = f"{source_code_fix_option.observation}\n{source_code_fix_option.how_to_fix}"
    output_parser = StrictPydanticOutputParser(pydantic_object=SourceCodeFix)
    prompt = load_prompt(get_prompt_file_path("fix_test_errors.yaml")).format(
        plan=plan,
        fixed_file=fixed_file_name,
        fixed_code=fixed_code,
        test_file=test_file_name,
        test_code=test_code,
        error_message=error_message,
        public_interface_document=public_interface_document.json(),
        specifications=specifications_text,
        format_instructions=output_parser.get_format_instructions(),
    )
    output = execute_model(model, prompt)
    source_code_fix = output_parser.parse(output)
    return source_code_fix


def apply_source_code_fix(source_code_fix: SourceCodeFix) -> None:
    with open(get_src_file_path(source_code_fix.file_name), "w") as f:
        f.write(source_code_fix.code)

    with open(get_doc_file_path(TEST_LOG_FILE_NAME), "a") as f:
        f.writelines(
            [
                f"\nFixed {source_code_fix.file_name}.\n",
                f"\n```\n{source_code_fix.code}\n```\n",
                "".join(["*" for _ in range(80)]),
                "\n",
            ]
        )

    logging.info(f"Fixed {source_code_fix.file_name}.")
