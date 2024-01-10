import langchain.chat_models as chat_models
import langchain.llms as llms

from templated._utils.chat2vanilla_lm import Chat2VanillaLM

LLM_CONSTRUCTORS = {
    **{name: Chat2VanillaLM(getattr(chat_models, name)) for name in chat_models.__all__},
    **{name: getattr(llms, name) for name in llms.__all__},
}

_DEFAULT_LLM = None


def make_default_llm(llm_kwargs={}):
    """
    Creates a default language model to use in templated.

    Args:
        llm_kwargs (dict, optional): Additional arguments to pass to the language model constructor. Defaults to {}.

    Returns:
        callable: A function that takes in a string prompt and returns a response.

    Raises:
        ValueError: If an invalid language model is specified.

    Example:
        >>> lm = make_default_llm()
        >>> type(lm("Hello, world!")) == str
        True
    """
    global _DEFAULT_LLM
    if _DEFAULT_LLM is not None:
        return _DEFAULT_LLM

    if "model" in llm_kwargs and llm_kwargs["model"] in LLM_CONSTRUCTORS:
        try:
            _DEFAULT_LLM = LLM_CONSTRUCTORS[llm_kwargs["model"]](**llm_kwargs)
            return _DEFAULT_LLM
        except Exception as e:
            raise ValueError(f'Invalid LLM model: {llm_kwargs["model"]}') from e
    for model in LLM_CONSTRUCTORS:
        try:
            _DEFAULT_LLM = LLM_CONSTRUCTORS[model](**llm_kwargs)
            return _DEFAULT_LLM
        except Exception as e:
            pass
    raise ValueError(f"No valid LLM model found")
