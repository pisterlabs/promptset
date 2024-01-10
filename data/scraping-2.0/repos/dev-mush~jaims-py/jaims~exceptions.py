class JAImsTokensLimitExceeded(Exception):
    """
    Exception raised when the token limit is exceeded.

    Attributes:
        max_tokens -- maximum number of tokens allowed
        messages_tokens -- number of tokens in the messages
        llm_buffer -- buffer for the LLM answer
        has_optimized -- flag indicating if the messages have been optimized
    """

    def __init__(self, max_tokens, messages_tokens, llm_buffer, has_optimized):
        message = f"Max tokens: {max_tokens}\n LLM Answer Buffer: {llm_buffer}\n Messages tokens: {messages_tokens}\n Messages Optimized: {has_optimized}  "
        super().__init__(message)


class JAImsMissingOpenaiAPIKeyException(Exception):
    """
    Exception raised when the OPENAI_API_KEY is missing.
    """

    def __init__(self):
        message = "Missing OPENAI_API_KEY, set environment variable OPENAI_API_KEY or pass it as a parameter to the agent constructor."
        super().__init__(message)


class JAImsOpenAIErrorException(Exception):
    """
    Exception raised when there is an error with OpenAI.

    Attributes:
        message -- explanation of the error
        openai_error -- the error from OpenAI
    """

    def __init__(self, message, openai_error):
        super().__init__(message)
        self.openai_error = openai_error


class JAImsMaxConsecutiveFunctionCallsExceeded(Exception):
    """
    Exception raised when the maximum number of consecutive function calls is exceeded.

    Attributes:
        max_consecutive_calls -- maximum number of consecutive calls allowed
    """

    def __init__(self, max_consecutive_calls):
        message = f"Max consecutive function calls exceeded: {max_consecutive_calls}"
        super().__init__(message)


class JAImsUnexpectedFunctionCall(Exception):
    """
    Exception raised when an unexpected function call occurs.

    Attributes:
        func_name -- name of the unexpected function
    """

    def __init__(self, func_name):
        message = f"Unexpected function call: {func_name}"
        super().__init__(message)
