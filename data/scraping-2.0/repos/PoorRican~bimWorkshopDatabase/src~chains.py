from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

_ROOT = Path(__file__).parent.parent

PARAMETER_PROMPT_FILE = _ROOT.joinpath("PARAMETER_PROMPT.txt")

PARAMETER_PROMPT = HumanMessagePromptTemplate.from_template(
    open(PARAMETER_PROMPT_FILE, "r").read()
)

VALUE_PROMPT_FILE = _ROOT.joinpath("VALUE_PROMPT.txt")

VALUE_PROMPT = HumanMessagePromptTemplate.from_template(
    open(VALUE_PROMPT_FILE, "r").read()
)


FORMAT_PROMPT = PromptTemplate.from_template(
    """Please format the given list as a valid python list:

For example:
1. foo: a description
2. bar: another description

becomes:
```python
["foo", "bar"]
```

Here is the list of values:
{content}

```python
[
""")


def extract_list_from_response(response: str) -> list[str]:
    """ Extract a list of values from the response.

    Parameters
    ----------
    response: str
        The response to extract values from. It should be a valid python list.

    Returns
    -------
    list[str]
        The list of values.
    """
    # remove any text after "]"
    response = response.split("]")[0]

    if response[0] != "[":
        response = "[" + response

    if response[-1] != "]":
        response = response + "]"

    extracted: list[str] = eval(response)
    return extracted


def build_formatter_chain(chat: ChatOpenAI) -> Runnable:
    """ Build a chain of runnables to format a list of values.

    The runnable accepts a dictionary with the following keys:
    - content: The list of values to format.

    The output of the runnable is a formatted string.

    Parameters
    ----------
    chat : ChatOpenAI
        The chatbot to use to generate values.

    Returns
    -------
    Runnable
        The chain of runnables to generate values.
    """
    return (
        FORMAT_PROMPT
        | chat
        | StrOutputParser()
    )


def build_parameter_chain(chat: ChatOpenAI) -> Runnable:
    """ Build a chain of runnables to generate a list of parameters for a given product.

    Parameters
    ----------
    chat : ChatOpenAI
        The chatbot to use to generate parameters.

    Returns
    -------
    Runnable
    """
    _prompt = ChatPromptTemplate.from_messages([PARAMETER_PROMPT])

    return (
        _prompt
        | chat
    )


def build_parameter_value_chain(chat: ChatOpenAI, parse_chat: ChatOpenAI) -> Runnable:
    """ Build a chain of runnables to generate a list of values for a given parameter.

    The runnable accepts a dictionary with the following keys:
    - parameter: The name of the parameter to generate values for.
    - product: The label of the product to generate values for.

    The output of the runnable is a parsed `ParameterList` object.

    Parameters
    ----------
    chat : ChatOpenAI
        The chatbot to use to generate values.
    parse_chat : ChatOpenAI
        The chatbot to use to parse the values. Should be of  lower temperature than `chat` and does not
        need to be as capable.

    Returns
    -------
    Runnable
        The chain of runnables to generate values.
    """
    value_prompt_messages = ChatPromptTemplate.from_messages([
        PARAMETER_PROMPT,
        MessagesPlaceholder(variable_name='ai_message'),
        VALUE_PROMPT])

    chain = (
        value_prompt_messages
        | chat
        | StrOutputParser()
    )

    formatter_chain = build_formatter_chain(parse_chat)

    return (
        {'content': chain}
        | formatter_chain
    )
