"""Support tools for embedchain."""

from os import makedirs
from hashlib import md5

from ruamel.yaml import YAML
import langchain
from langchain.schema import SystemMessage
from langchain.cache import SQLiteCache
from embedchain import App
from embedchain.config import ChromaDbConfig, LlmConfig

from hivemind.config import EMBEDCHAIN_DATA_DIR, LANGCHAIN_CACHE_DIR
from hivemind.toolkit.models import query_model, precise_model
from hivemind.toolkit.text_extraction import extract_blocks

yaml = YAML()
langchain.llm_cache = SQLiteCache(
    database_path=str(LANGCHAIN_CACHE_DIR / ".langchain.db")
)


def query_resource(resource_location: str, query: str, update: bool = False) -> str:
    """Ask a question about a resource."""
    resource_hash = md5(resource_location.encode()).hexdigest()
    # hack to restrict query to use only the resource's embeddings
    resource_dir = EMBEDCHAIN_DATA_DIR / resource_hash
    new_resource = not resource_dir.exists()
    makedirs(resource_dir, exist_ok=True)
    qna_bot = App(db_config=ChromaDbConfig(dir=str(resource_dir)))
    if new_resource or update:
        qna_bot.add(resource_location)
    return qna_bot.query(query, config=LlmConfig(model="gpt-4"))


def test_query_resource() -> None:
    """Test query_resource."""

    # expect positive answer of info post knowledge-cutoff
    response = query_resource(
        "https://en.wikipedia.org/wiki/OpenAI",
        "Tell me about the recent history of OpenAI.",
    )
    print(f"{response=}")

    # expect positive answer
    response_2 = query_resource(
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Tell me about the philosophy of Python.",
    )
    print(f"{response_2=}")

    # expect negative answer since this is the wrong resource for the question
    response_3 = query_resource(
        "https://en.wikipedia.org/wiki/OpenAI",
        "Tell me about the philosophy of Python.",
    )
    print(f"{response_3=}")


def validate(message: str) -> str | None:
    """Check for missing components in messages that are meant for conversion to `query_resource` params. Error is meant to be sent back to a user (either human or agent)."""
    instructions = """
    You are a message validation bot. Your purpose is to check for specific components that must be present in a message, and to return an error message if they are missing.

    The message you received is:
    ```text
    {message}
    ```

    The message is meant to be a query to a particular resource, and must contain the following components:
    - the location of the resource, as a URI
    - the query (either a question or a topic)

    Output your validation result as a markdown `text` block. If the message has no errors, output the following:
    ```text
    N/A
    ```
    If the message has errors, output the following (fill in the error message):
    ```text
    Your request was invalid due to missing required components.
    Error: <error>
    Original Message: "{message}"
    ```
    """
    instructions = instructions.format(message=message)
    messages = [SystemMessage(content=instructions)]
    result = query_model(precise_model, messages, printout=False)
    error = extract_blocks(result, "text")
    if not error or ("N/A" not in error[-1] and "Error:" not in error[-1]):
        raise ValueError(f"Unable to extract error validation result:\n{error}")
    error_text = error[-1].strip()
    return error_text if "Error:" in error_text else None


def test_validate() -> None:
    """Test check_for_error."""
    message = "Tell me about the philosophy of Python."
    error = validate(message)
    print(f"{error=}")  # expect error, since we're missing the resource location
    message = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    error = validate(message)
    print(f"{error=}")  # expect error, since we're missing the query
    message = "Tell me about the philosophy of Python from the page at https://en.wikipedia.org/wiki/Python_(programming_language)"
    error = validate(message)
    print(
        f"{error=}"
    )  # expect no error, since we have both the resource location and the query


if __name__ == "__main__":
    # test_check_for_error()
    # test_query_resource()
    pass
