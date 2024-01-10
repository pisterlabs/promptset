import logging
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from parsers.strict_pydantic_output_parser import StrictPydanticOutputParser
from schema import File, PublicInterfaceDocument
from util import (
    PUBLIC_INTERFACE_DOCUMENT_NAME,
    execute_model,
    get_acceptance_test_file_name,
    get_doc_file_path,
    get_prompt_file_path,
    get_src_file_path,
)


def generate_public_interface_document(
    model: ChatOpenAI,
    specifications_text: str,
) -> PublicInterfaceDocument:
    doc_file_path = get_doc_file_path(PUBLIC_INTERFACE_DOCUMENT_NAME)
    if os.path.exists(doc_file_path):
        logging.info(f"Reusing {doc_file_path}.")
        return PublicInterfaceDocument.parse_file(doc_file_path)

    logging.info("Generating public interface document.")

    output_parser = StrictPydanticOutputParser(pydantic_object=PublicInterfaceDocument)
    prompt = load_prompt(
        get_prompt_file_path("gen_public_interface_document.yaml")
    ).format(
        specifications=specifications_text,
        format_instructions=output_parser.get_format_instructions(),
    )
    output = execute_model(model, prompt)
    public_interface_document = output_parser.parse(output)

    with open(get_doc_file_path(PUBLIC_INTERFACE_DOCUMENT_NAME), "w") as f:
        f.write(public_interface_document.json())

    return public_interface_document


def update_public_interface_document(
    model: ChatOpenAI,
    public_interface_document: PublicInterfaceDocument,
    file_names: list[str] = None,
    force: bool = False,
) -> PublicInterfaceDocument:
    acceptance_test_file_name = get_acceptance_test_file_name(
        0, public_interface_document.entry_point_file_name
    )
    if os.path.exists(get_src_file_path(acceptance_test_file_name)) and not force:
        logging.info(
            f"Skipping public interface document update because "
            f"{acceptance_test_file_name} exists."
        )
        return public_interface_document

    files = (
        public_interface_document.files
        if file_names is None
        else [
            file for file in public_interface_document.files if file.name in file_names
        ]
    )

    for file in files:
        logging.info(f"Updating public interface document for {file.name}.")

        with open(get_src_file_path(file.name)) as f:
            source_code = f.read()

        output_parser = StrictPydanticOutputParser(pydantic_object=File)
        prompt = load_prompt(
            get_prompt_file_path("update_public_interface_document.yaml")
        ).format(
            public_interface_document=file.json(),
            file=file.name,
            source_code=source_code,
            format_instructions=output_parser.get_format_instructions(),
        )
        output = execute_model(model, prompt)
        updated_file = output_parser.parse(output)

        public_interface_document.files = [
            updated_file if f.name == updated_file.name else f
            for f in public_interface_document.files
        ]

    with open(get_doc_file_path(PUBLIC_INTERFACE_DOCUMENT_NAME), "w") as f:
        f.write(public_interface_document.json())

    return public_interface_document
