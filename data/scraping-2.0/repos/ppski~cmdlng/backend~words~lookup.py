import spacy

from typing import Union
from django.conf import settings

from .models import Word
from .cmd_display import WordDisplay
from .look_up_sources._diclookup import DictionaryLookUp
from .look_up_sources.lexicala import Lexicala
from .look_up_sources.wordreference import WordReference
from .look_up_sources.llm_openai import OpenAI
from .look_up_sources.llm_langchain import LangChain


# Look for word in db
class WordSearch:
    def __init__(
        self,
        lookup_word: str,
        lang_source: str = settings.LANG_SOURCE,
        lang_target: str = settings.LANG_TARGET,
    ):
        self.lookup_word = lookup_word
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.lemma = self.get_spacy_lemma()

    def get_spacy_lemma(self):
        if self.lang_source == "fr_fr":
            nlp = spacy.load("fr_core_news_sm")
        elif self.lang_source == "en_us":
            nlp = spacy.load("en_core_web_sm")
        else:
            # Use EN for other languages
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(self.lookup_word)
        return doc[0].lemma_  # FIXME: MWEs

    def search(self, lookup_word=None) -> Union[Word, None]:
        # Search for a word in the db
        search_lookup_as_lemma_result = Word.objects.filter(
            lemma=lookup_word, lang_source=self.lang_source
        )

        if search_lookup_as_lemma_result:
            return search_lookup_as_lemma_result

        search_lookup_result = Word.objects.filter(
            lookup_word=lookup_word, lang_source=self.lang_source
        )
        if search_lookup_result:
            return search_lookup_result

        # Get lemma
        spacy_lemma = self.get_spacy_lemma()
        search_lemma_result = Word.objects.filter(
            lemma=spacy_lemma, lang_source=self.lang_source
        )
        if search_lemma_result:
            return search_lemma_result

        return None


# Look up word in APIs
class WordLookUp:
    def __init__(
        self,
        lookup_word: str,
        lemma: str,
        lang_source: str = "fr_fr",
        lang_target: str = "fr_fr",
        llm: Union[bool, str] = False,
    ):
        self.lookup_word = lookup_word
        self.lang_source = lang_source
        self.lang_prefix = lang_source[:2]  # fr_fr -> fr
        self.lang_target = lang_target
        self.lang_pair = f"{lang_source}-{lang_target}"
        self.lemma = lemma
        self.llm = llm

    def look_up(self) -> Union[str, None]:
        results = self.get_search_pref()

        if not results:
            return None

        for result in results:
            empty_dict = {
                "lang_source": self.lang_source,
                "lang_target": self.lang_target,
                "lookup_word": self.lookup_word,
                "lemma": self.lemma,  # This is the spacy lemma
                "related_lemma": result.get("related_lemma"),
                "native_alpha_lemma": result.get("native_alpha_lemma"),
                "pos": DictionaryLookUp.clean_lookup_pos(result.get("pos")),
                "en_translation": result.get("en_translation"),
                "definition": result.get("definition"),
                "source": result.get("source"),
                "examples": result.get("examples"),
                "pos_forms": result.get("pos_forms"),
                "is_mwe": False,  # FIXME
                "is_informal": False,  # FIXME
            }
            new_word = Word(**empty_dict)
            new_word.save()
        new_entry = Word.objects.filter(
            lemma=self.lemma, lang_source=self.lang_source
        ).first()
        return new_entry

    def get_search_pref(self) -> Union[list, None]:
        preferred_results = None
        if self.llm and self.llm == "chatgpt":
            preferred_results = OpenAI.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
        elif self.llm and self.llm == "llama":
            preferred_results = LangChain.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
        elif self.lang_pair == "fr_fr-fr_fr":
            preferred_results = Lexicala.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
        elif self.lang_pair == "fr_fr-en_us":
            preferred_results = Lexicala.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
        elif self.lang_pair == "it_it-en_us":
            preferred_results = Lexicala.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
        if not preferred_results:
            default_results = WordReference.look_up(
                self.lookup_word,
                self.lemma,
                self.lang_prefix,
                self.lang_source,
                self.lang_target,
            )
            return default_results

        return preferred_results
