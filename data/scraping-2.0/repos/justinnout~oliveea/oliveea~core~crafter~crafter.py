from langchain import LLMChain


class Crafter:
    """
    This module, Craft, is designed to enhance a string of words into another string of words in a given tone.
    It utilizes a Language Model (LLM) to achieve this transformation. The enhancement process involves
    adjusting the original string to match the desired tone while maintaining the semantic meaning of the input.

    Attributes:
        None

    Methods:
        __init__ : Initializes the Crafter class.
        craft  : Crafts the input string of words to match the desired tone.
    """

    def __init__(self, llm, prompt):
        """
        The constructor for Crafter class.
        """
        self.llm_chain = LLMChain(
            prompt=prompt,
            llm=llm,
        )

    def craft(self, words: str, tone) -> str:
        """
        The function to enhance the input string of words to match the desired tone.

        Parameters:
            words (str): The original string of words to be enhanced.
            tone : The desired tone to which the original string should be enhanced.

        Returns:
            None: The function is yet to be implemented.
        """
        return self.llm_chain.run(input=words, tone=tone)
