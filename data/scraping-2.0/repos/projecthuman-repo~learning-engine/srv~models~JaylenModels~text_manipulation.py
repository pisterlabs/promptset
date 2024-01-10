"""
author: Jay Edwards

This class handles extracting keywords and paragraphs from a text.
"""

from collections import Counter
import language_tool_python as ltp
import openai
import spacy


class TextManipulation:
    """
    This class provides methods for extracting information from a given text.

    Dependencies:
    - Counter
    - openai
    - spacy
    - language_tool_python


    Attributes:
    -----------
    - nlp: loads the english tokenizer for spacy so that it can parse the language
    - tool: load the enlgish language tool for automatic correction

    Methods:
    --------
    get_keywords(text: str, frequency: int = 1) -> List[str]:
        Extracts the keywords from the given text using OpenAI GPT-3 or Spacy NLP and returns a list of keywords.

    get_sentences(text: str) -> List[str]:
        Extracts the sentences from the given text using Spacy NLP and returns a list of sentences.

    get_paragraphs(text: str) -> List[str]:
        Extracts the paragraphs from the given text using OpenAI GPT-3 or simple newline splitting and returns a list of paragraphs.

    check_text(text: str) -> str:
        Corrects the grammar and spelling errors in the given text using LanguageTool and returns the corrected text.

    gpt_query(query: str) -> Union[None, OpenAICompletion]:
        Sends the given query to OpenAI GPT-3 and returns the response.

    """
    # Download and load the spacy tokenizer for english
    def __init__(self):
        # Download English tokenizer
        spacy.cli.download("en_core_web_sm")
        # Load English tokenizer, tagger, parser and NER
        self.nlp = spacy.load("en_core_web_sm")
        # use a local server (automatically set up), language English
        self.tool = ltp.LanguageTool('en-US')
        # Get open AI key for chat gpt
        openai.api_key = "OPENAI_KEY"

    # Return a list of keywords found in the text
    def get_keywords(self, text="", frequency=1):
        query = "Find the keywords in this text: " + text
        response = self.gpt_query(query)  # query gpt for the above request

        if response is not None:  # if returns something it almost always right so return
            return response
        else:  # otherwise use our own method
            initial_keys = []
            doc = self.nlp(text)

            # find the entities in the text that are NOT text
            for entity in doc.ents:
                if str(entity.text).isalpha() and len(entity.text) > 2:
                    initial_keys.append(entity.text)

            # find the tokens in the text that are pronouns, adjectives and nouns (these are good candidates for keys)
            # temporarily blocked off unless anyone finds it useful
            if len(initial_keys) == 0:
                for token in doc:
                    if token.text in ['PROPN', 'ADJ', 'NOUN']:
                        initial_keys.append(token.text)

            # remove repeat elements from the initial keywords
            initial_keys = sorted(list(set(initial_keys)))

            # this is for returning keywords when the frequency for keywords is higher than 1
            if frequency > 1:
                keywords = []
                word_frequencies = Counter(text.split()).items()
                for word in word_frequencies:
                    if word[0] in initial_keys and word[1] > frequency:
                        keywords.append(word[0])
                return keywords
            # if no frequencies are returned, just use a default
            return initial_keys

    # Return a list of sentences detected
    def get_sentences(self, text=""):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        return sentences

    # This function is quite simple, determine a list of paragraphs by splitting at  newlines
    # This needs to be better in the future as it may and certainly WILL cause issues. For now it will suffice.
    # Note: optimized with chat cpt. Still need to work out the kinks
    def get_paragraphs(self, text=""):
        query = "Identify the paragraphs in this text: " + text
        response = self.gpt_query(query)
        # if the above does not work do a cheap method
        if response is None or "":
            return text.split('\n')
        # if it does work return the answer
        return response

    # this method corrects grammatical and spelling errors.
    # will be useful for the questgen outputs.
    def check_text(self, text=""):
        matches = self.tool.check(text)
        return ltp.utils.correct(text, matches)

    # this method sends queries to chat gpt works awful though and often gives errors.
    def gpt_query(self, query=""):
        try:
            # send above query to gpt
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=query,
                temperature=0.7,
                max_tokens=709,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
        except openai.error.RateLimitError:
            return None
        return response
