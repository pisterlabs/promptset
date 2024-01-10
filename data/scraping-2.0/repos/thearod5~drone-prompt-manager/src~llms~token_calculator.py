import tiktoken

from llms.llm_models import OpenAIModel

TOKENS_2_WORDS_CONVERSION = (3 / 4)  # open ai's rule of thumb for approximating tokens from number of words
MAX_TOKENS_BUFFER = 400
MAX_TOKENS_DEFAULT = 2000


class TokenCalculator:

    @staticmethod
    def calculate_max_tokens(model: OpenAIModel, prompt: str) -> int:
        """
        Gets the token limit for the given model with the given max tokens for completion
        :param model: The model used to predict on prompt.
        :param prompt: The prompt being given to model for completion.
        :return: The max token allowed for given model and prompt.
        """
        model_token_limit = model.get_max_tokens()
        prompt_tokens = TokenCalculator.estimate_num_tokens(prompt, model)
        return model_token_limit - prompt_tokens - MAX_TOKENS_BUFFER

    @staticmethod
    def estimate_num_tokens(content: str, model: OpenAIModel) -> int:
        """
        Approximates the number of tokens that some content will be tokenized into by a given model by trying to tokenize
            and giving a rough estimate using a words to tokens conversion if that fails
        :param content: The content to be tokenized
        :param model: The model that will be doing the tokenization.
        :return: The approximate number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model.value)
            num_tokens = len(encoding.encode(content))
            return num_tokens
        except Exception:
            return TokenCalculator.rough_estimate_num_tokens(content)

    @staticmethod
    def rough_estimate_num_tokens(content: str) -> int:
        """
        Gives a rough estimate the number of tokens that some content will be tokenized into using the 4/3 rule used by open ai
        :param content: The content to be tokenized
        :return: The approximate number of tokens
        """
        return round(len(content.split()) * (1 / TOKENS_2_WORDS_CONVERSION))
