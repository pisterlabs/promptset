from typing import Any, Dict

import langchain.utilities.opaqueprompts as op
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BasePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnableSequence
from opchatserver.models import ChatResponse


def get_intermediate_output_chain(
    prompt: ChatPromptTemplate, llm: LLM
) -> RunnableSequence:
    """
    Build and return a chain that can give intermediate outputs, using
    the LangChain expression language. It uses sanitize() and desanitize()
    from the OpaquePrompts functions to avoid leaking senitive information
    to the llm.

    This is used by the chat server to get intermediate outputs, which is
    not a general use case of OpaquePrompts.
    For a simpler usage of OpaquePrompts, please see `OpaquePromptsLLMWrapper`.

    Parameters
    ----------
    prompt : ChatPromptTemplate
        the prompt template used by the chain
    llm : LLM
        the llm used by the chain

    Returns
    -------
    RunnableSequence
        the chain that can give intermediate outputs including
        `sanitizedPrompt`, `rawResponse`, and `desanitizedResponse`.

        `sanitizedPrompt` is the prompt after sanitization.
        `rawResponse` is the raw response from the llm.
        `desanitizedResponse` is the response after desanitization.
    """
    pg_chain: RunnableSequence = (
        RunnableMap(
            {
                # sanitize the input
                "inputs_after_sanitize": (_sanitize),
            }
        )
        | RunnableMap(
            {
                # get the sanitized prompt
                "sanitized_prompt": (
                    lambda x: x["inputs_after_sanitize"]["sanitized_input"][
                        "prompt"
                    ]
                ),
                # pass the sanitized input to the llm and get the raw response
                "raw_response": (
                    lambda x: x["inputs_after_sanitize"]["sanitized_input"]
                )
                | prompt
                | llm
                | StrOutputParser(),
                # pass through the secure context from the sanitized input
                "secure_context": (
                    lambda x: x["inputs_after_sanitize"]["secure_context"]
                ),
            }
        )
        | RunnableMap(
            {
                "sanitizedPrompt": (lambda x: x["sanitized_prompt"]),
                "rawResponse": (lambda x: x["raw_response"]),
                # desanitize the response
                "desanitizedResponse": (
                    lambda x: op.desanitize(
                        x["raw_response"],
                        x["secure_context"],
                    )
                ),
            }
        )
    )
    return pg_chain


def get_response(
    prompt: BasePromptTemplate,
    memory: ConversationBufferWindowMemory,
    input: str,
    llm: LLM,
) -> ChatResponse:
    """
    Get chat response with intermediate outputs

    Parameters
    ----------
    prompt : BasePromptTemplate
        the prompt template used by the chain
    memory : ConversationBufferWindowMemory
        memory that stores the conversation history
    input : str
        the user input message
    llm : LLM
        the llm used by the chain

    Returns
    -------
    ChatResponse
        the chat response with intermediate outputs including
        `sanitizedPrompt`, `rawResponse`, and `desanitizedResponse`.

        `sanitizedPrompt` is the prompt after sanitization.
        `rawResponse` is the raw response from the llm.
        `desanitizedResponse` is the response after desanitization.
    """
    pg_chain = get_intermediate_output_chain(prompt, llm=llm)
    return ChatResponse(
        **pg_chain.invoke(
            {
                "prompt": input,
                "history": memory.buffer_as_messages,
            }
        )
    )


def _sanitize(unsanitized_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function which splits up the history part of the prompt prior to
    sanitizing it in order to improve the OpaquePrompts sanitize functions
    performance. The function then combines the history field back together.

    This function wraps the call to op.sanitize().

    Parameters
    ----------
    unsanitized_input : dict
        The unsanitized input that needs to be sanitized. If it does not
        contain the "history" key, it will just be passed directly to
        op.sanitize. If the history key is contained in the dictionary then
        it should contain a list of alternating HumanMessage and AIMessage
        objects and even length.

    Returns
    -------
    dict
        Sanitized input dict of strings and the secure context
        as a dict following the format:
        {
            "sanitized_input": <sanitized form of unsanitized_input (dict)>,
            "secure_context": <secure context (str)>
        }

        The `secure_context` needs to be passed to the `desanitize` function.
    """
    if "history" not in unsanitized_input:
        return op.sanitize(unsanitized_input)
    # Convert history from an array of chat messages to a dictionary
    history = unsanitized_input.pop("history")
    split_history = [message.content for message in history]
    if len(split_history) % 2 != 0:
        raise Exception("History must be an even length string")
    for i, chat_message in enumerate(split_history):
        prefix = "Human"
        if i % 2 == 1:
            prefix = "Ai"
        unsanitized_input[f"{prefix} {i//2 + 1}"] = chat_message
    sanitized_response = op.sanitize(unsanitized_input)
    # Reconstruct original history str with the sanitized messages
    sanitized_input = sanitized_response["sanitized_input"]

    history_str = ""
    for i in range(1, len(split_history) // 2 + 1):
        human_message = sanitized_input.pop(f"Human {i}")
        ai_message = sanitized_input.pop(f"Ai {i}")
        history_str += f"Human: {human_message}AI: {ai_message}"
    sanitized_input["history"] = history_str
    return sanitized_response
