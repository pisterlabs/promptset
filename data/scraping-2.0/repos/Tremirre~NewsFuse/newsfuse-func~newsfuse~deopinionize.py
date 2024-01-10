import logging

import openai


def merge_sentences(sentences: list[str]) -> str:
    """
    Merges a list of sentences into a single string.
    Uses a newline character as a separator.

    :param sentences: List of sentences to merge
    :return: Merged sentences
    """
    return "\n".join(sentences)


class OpinionRemover:
    """
    Class for removing opinions from a text using OpenAI's API.
    """

    def __init__(
        self,
        translation_key: str,
        task: str,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        """
        :param translation_key: key for OpenAI API
        :param task: detailed description of the task to be sent to OpenAI API
        :param model: selected model for generating deopinionized sentences,
            defaults to "gpt-3.5-turbo"
        """
        self.model = model
        self.task = task
        openai.api_key = translation_key

    def remove_opinions(self, sentences: list[str]) -> dict | None:
        """
        Removes opinions from a list of sentences.
        If the request fails, returns None.

        :param sentences: List of sentences to remove opinions from
        :return: Deopinionized sentences
        """
        if not sentences:
            return None
        corpus = merge_sentences(sentences)
        try:
            return self.send_request(corpus)
        except openai.OpenAIError as e:
            logging.error("Failed to send request to OpenAI: " + str(e))
            return None

    def send_request(self, content: str) -> dict:
        """
        Sends a deopnionize request to OpenAI API.

        :param content: text to be deopinionized sentence by sentence
        :return: raw response from OpenAI API
        """
        return openai.ChatCompletion.create(  # type: ignore
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.task,
                },
                {"role": "user", "content": content},
            ],
        )
