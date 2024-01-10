"""

This module contains the completion class.

"""

# Import from standard library
import os
import logging
from dotenv import load_dotenv


# Import from 3rd party libraries
import cohere
import streamlit as st


# Configure logger
logging.getLogger("complete").setLevel(logging.WARNING)


# Load environment variables
load_dotenv()


# Assign credentials from environment variable or streamlit secrets dict
co = cohere.Client(os.getenv("COHERE_API_KEY")) or st.secrets["COHERE_API_KEY"]


class Completion:

    def ___init___(self, ):
        pass

    @staticmethod
    def complete(prompt, max_tokens, temperature, stop_sequences):
      
        """
        Call Cohere Completion with text prompt.
        Args:
            prompt: text prompt
            max_tokens: max number of tokens to generate
            temperature: temperature for generation
            stop_sequences: list of sequences to stop generation
        Return: predicted response text
        """
       
        try:
            response = co.generate(  
                model = 'xlarge',
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature,
                stop_sequences = stop_sequences)
            
            print("generated text:\n", response.generations[0].text)
            return response.generations[0].text

        
        except Exception as e:
            logging.error(f"Cohere API error: {e}")
            st.session_state.text_error = f"Cohere API error: {e}"
            print("Error:", e)


# Usage

# c = Completion()
# r = c.complete("I want to play", 40, 0.8, ["--"])
# print(r)
