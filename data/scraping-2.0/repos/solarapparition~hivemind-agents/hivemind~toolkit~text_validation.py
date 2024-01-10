"""Text validation tools."""
from typing import Sequence

from langchain.schema import SystemMessage, BaseMessage

from hivemind.toolkit.models import query_model, precise_model
from hivemind.toolkit.text_formatting import dedent_and_strip, extract_blocks


def find_llm_validation_error(validation_messages: Sequence[BaseMessage]) -> str | None:
    """Run validation using an LLM. A specific output format is expected:
    If the message has no errors:
    ```text
    N/A
    ```

    If the message has errors:
    ```text
    <error>
    ```
    Where <error> contains the string "Error:".
    """
    result = query_model(precise_model, validation_messages, printout=False)
    error = extract_blocks(result, "text")
    error_flag = "ERROR:"
    if not error or ("N/A" not in error[-1] and error_flag not in error[-1]):
        raise ValueError(f"Unable to extract error validation result:\n{error}")
    error_text = error[-1].strip()
    return error_text if error_flag in error_text else None


def validate_text(
    text: str, requirements: str, context: str | None = None
) -> tuple[bool, str]:
    """Validate text output based on freeform requirements. Requirements should be relatively simple—something that a human who understands the context could check in a few seconds."""
    instructions = """
    # MISSION
    You are a text validation bot. Your purpose is to validate that the text that you're given meets certain requirements.
    
    # VALIDATION CONTEXT
    Here is some context in which the text is being evaluated to help with validation:
    ```text
    {context}
    ```

    # TEXT REQUIREMENTS
    Here are the requirements to check for the text you're given:
    ```text
    {requirements}
    ```

    # INPUT
    The text you're given is:
    ```text
    {text}
    ```
    
    # OUTPUT
    Check whether the text meets the requirements.

    If the text meets the requirements, output the following (include the backtick delimiters):
    ```text
    N/A
    ```

    If the text does not meet the requirements, output the following (fill in the error message):
    ```text
    ERROR: {{error}}
    ```
    You may output other comments, but all other comments must be outside the ```text``` block.
    """
    instructions = dedent_and_strip(instructions).format(
        text=text, requirements=requirements, context=context
    )
    error = find_llm_validation_error([SystemMessage(content=instructions)])
    error_message = dedent_and_strip(
        """
        {error}
        Original Text: "{text}"
        """
    ).format(error=error, text=text)
    return error is None, "" if error is None else error_message
