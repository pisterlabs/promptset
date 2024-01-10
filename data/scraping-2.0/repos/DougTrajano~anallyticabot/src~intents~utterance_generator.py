import openai
import streamlit as st
from src.helper_functions import setup_logger

logger = setup_logger()


class UtteranceGenerator:
    def __init__(self, openai_api_key: str, engine: str = "davinci",
                 temperature: float = 0.7, max_tokens: int = 32,
                 top_p: float = 1.0, freq_p: float = 0.2,
                 pres_p: float = 0.2, **kwargs):
        """
        Utterance Generator class.

        Args:
            openai_api_key (str): OpenAI API key.
            engine (str): Name of the text-generating engine.
            temperature (float): Temperature of the text-generating engine.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Probability of generating a top-N candidate.
            freq_p (float): Probability of generating a frequent candidate.
            pres_p (float): Probability of generating a present candidate.
        """        

        logger.info(f"Initializing utterance generator with {engine}")

        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.freq_p = freq_p
        self.pres_p = pres_p

        openai.api_key = openai_api_key

    def generate(self, prompt: str, qty: int = 1, increment: bool = False) -> list:
        """
        Generate a list of utterances from a prompt.

        Args:
            prompt (str): The prompt to use for the utterance generation.
            qty (int): The number of utterances to generate.
            increment (bool): If True, the prompt will be incremented by the last generated utterance.

        Returns:
            A list of utterances.
        """
        new_utterances = []

        pbar_counter = 0
        pbar = st.progress(pbar_counter)

        for _ in range(qty):
            response = self._call_openai_gpt(prompt)

            for row in response["choices"]:
                new_utterances.append(row["text"])
                if increment:
                    prompt = prompt + "\n" + row["text"]

            pbar_counter += 1
            pbar.progress(pbar_counter / qty)
            
        return new_utterances

    def _call_openai_gpt(self, prompt: str) -> dict:
        """
        Call the OpenAI GPT text-generating engine.
        
        Args:
            prompt (str): The prompt to use for the utterance generation.

        Returns:
            A dictionary with the response from OpenAI GPT.
        """
        logger.info(f"Calling OpenAI GPT-3 with prompt: {prompt}")
            
        if prompt[-1:] != "\n":
            prompt += "\n"

        response = openai.Completion.create(engine=self.engine,
                                            prompt=prompt,
                                            temperature=self.temperature,
                                            max_tokens=self.max_tokens,
                                            top_p=self.top_p,
                                            frequency_penalty=self.freq_p,
                                            presence_penalty=self.pres_p,
                                            stop="\n")

        logger.info(f"OpenAI GPT-3 response: {response}")
        return response
