"""
Other modules: Summarize, Change of Tone, Rephrase
"""
from langchain.chains import LLMChain
from src.prompts.prompts_template import (
    CHANGE_TONE_CHAIN,
    SUMMARIZE_CHAIN,
    REPHRASE_CHAIN,
    PROMPT_PARRAGRAPH_SUGGESTION
)

# Classes definition for Exception Handling & Identification
class SummarizeException(Exception):
    """Custom Exception for Summarization Function"""

class ChangeOfToneException(Exception):
    """Custom Exception for Change of Tone Funtion"""

class RephraseException(Exception):
    """Custom Exception for Rephrase Function"""

class ParagraphSuggestionException(Exception):
    """Custom Exception for Paragraph Suggestion Function"""



def summarize_text(text, llm, verbose: bool = False):
    """
    Summarize text
    """
    try:
        if text == None or len(text) ==0:
            raise SummarizeException(
                "Text to summarize cannot be empty."
            )

        summarize_chain = LLMChain(
            llm=llm, output_key="summary", verbose=verbose, prompt=SUMMARIZE_CHAIN
        )
        result = summarize_chain.run(text=text)
        print('Summary result: ', result)
        return result
    
    except Exception as error:
        raise SummarizeException(
            f"Exception caught in Summarization Module: {error}"
        )


def change_of_tone_text(text, tone_description, llm, verbose: bool = False):
    """
    Change of tone
    """
    try:
        
        if text == None or len(text) == 0 or tone_description == None or len(tone_description) == 0:
            raise ChangeOfToneException(
                "Text to change tone or tone description cannot be empty."
            )

        
        change_tone_chain = LLMChain(
            llm=llm, output_key="changed_tone_text", verbose=verbose, prompt=CHANGE_TONE_CHAIN
        )
        result = change_tone_chain.run(
            tone_description=tone_description, text=text)
        print('Change of tone result: ', result)
        return result
    
    except Exception as error:
        raise ChangeOfToneException(
            f"Exception caught in Change of Tone Module: {error}"
        )


def rephrase_text(text, llm, verbose: bool = False):
    """
    Rephrase
    """
    try:

        if text == None or len(text) == 0:
            raise RephraseException(
                "Text to rephrase cannot be empty."
            )

        rephrase_chain = LLMChain(
            llm=llm, output_key="pharaphrased_text", verbose=verbose, prompt=REPHRASE_CHAIN
        )
        result = rephrase_chain.run(text=text)
        print('Rephrase result: ', result)
        return result
    
    except Exception as error:
        raise RephraseException(
            f"Exception caught in Rephrase Module: {error}"
        )


def parragraph_suggestion(text: str, llm, verbose: bool = False) -> str:
    """
    Suggest a new paragraph
    """
    try:

        if text == None or len(text) == 0:
            ParagraphSuggestionException(
                "Input text for paragraph suggestion cannot be empty."
            )

        parragraph_suggestion_chain = LLMChain(
            llm=llm, output_key="suggestion", verbose=verbose, prompt=PROMPT_PARRAGRAPH_SUGGESTION
        )

        result = parragraph_suggestion_chain.run(context=text)
        print('Parragraph suggestion result: ', result)
        return result

    except Exception as error:
        raise ParagraphSuggestionException(
            f"Exception caught in Paragraph Suggestion Module: {error}"
        )
