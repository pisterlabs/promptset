import json
import openai

from django.conf import settings
from langchain.llms import CTransformers
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from typing import Union

from ._diclookup import DictionaryLookUp


class LangChain(DictionaryLookUp):
    @staticmethod
    def look_up(
        lookup_word: str,
        lemma: str,
        lang_prefix: str,
        lang_source: str,
        lang_target: str,
    ) -> Union[list, None]:
        print("LOOKING UP WORD WITH LANGCHAIN")

        model = CTransformers(
            model=settings.LOCAL_LLAMA_MODEL, model_type=settings.DEFAULT_LOCAL_LLM
        )

        response_schemas = [
            ResponseSchema(
                name="en_translation",
                description="English translation of the lookup_word",
            ),
            ResponseSchema(
                name="definition", description="Definition of the lookup_word"
            ),
            ResponseSchema(
                name="lemma", description="Lemma of the lookup_word in the target_lang"
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template='ASSISTANT: {format_instructions}\n\nUSER: Provide the translation, lemma (in {target_lang}), and definition of "{lookup_word}" from {lang_source} into {target_lang}\n\nASSISTANT: ',
            input_variables=["lookup_word", "target_lang", "lang_source"],
            partial_variables={"format_instructions": format_instructions},
        )

        _input = prompt.format_prompt(
            lookup_word="cannelle", target_lang="English", lang_source="French"
        )
        output = model(_input.to_string())

        try:
            result = output_parser.parse(output)
            result["llm"] = settings.DEFAULT_LOCAL_LLM_NAME
            print(result)
            return [result]
        except:
            print("No results")
            return None
