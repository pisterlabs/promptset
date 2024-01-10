import os
import openai
import time
import logging
from typing import Dict, Any, Optional, Tuple

from src.linkers.abstract_entity_linker import AbstractEntityLinker
from src.models.entity_database import EntityDatabase
from src.models.entity_prediction import EntityPrediction
from src.utils.knowledge_base_mapper import KnowledgeBaseMapper


logger = logging.getLogger("main." + __name__.split(".")[-1])


def get_text_after(offset, text, n_chars=30):
    if offset < 0 or offset > len(text):
        return ""
    return text[offset:min(offset + n_chars, len(text) - 1)]


class GPTLinker(AbstractEntityLinker):
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        self.model = None

        # Get config variables
        self.model_name = config["model"] if "model" in config else "gpt-4-1106-preview"
        self.linker_identifier = config["linker_name"] if "linker_name" in config else f"GPT ({self.model_name})"
        self.ner_identifier = self.linker_identifier
        openai.api_key = config["openai_api_key"] if "openai_api_key" in config else os.getenv("OPENAI_API_KEY")
        self.temperature = config["temperature"] if "temperature" in config else 0.5
        self.seed = config["seed"] if "seed" in config else 42
        self.named_only = config["named_only"] if "named_only" in config else True

    def predict(self,
                text: str,
                doc=None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        named_entity_instruction = "Annotate only named entities. " if self.named_only else ""
        instructions = "You are an excellent linguist. Annotate the text given by the user with Wikipedia entities. " \
                       "Your reply should consist only of the text given by the user and entity annotations within " \
                       "this text of the format \"[mention text]{Wikipedia entity name}\". Make sure to use curly " \
                       "brackets for the entity name, NOT round ones. The Wikipedia entity name is the last element " \
                       "of the Wikipedia URL for this entity, e.g. the Wikipedia URL for the footballer Michael " \
                       "Jordan is \"https://en.wikipedia.org/wiki/Michael_Jordan_(footballer)\" so the Wikipedia " \
                       "entity name is \"Michael_Jordan_(footballer)\". If you know something is an entity but you " \
                       "are unsure which Wikipedia entity to predict or you believe there is no matching Wikipedia " \
                       "entity, annotate the mention as \"[mention text]{}\". " \
                       + named_entity_instruction + \
                       "It is essential that you do not change the original text except for adding the entity " \
                       "annotations.\n"

        named_version_example = "Output: \"[Albert]{Albert_Einstein} was one of the greatest physicists of all time. " \
                                "He was born in [Ulm]{Ulm} in 1879.\"\n"
        non_named_version_example = "Output: \"[Albert]{Albert_Einstein} was one of the greatest " \
                                    "[physicists]{Physicist} of all time. He was born in [Ulm]{Ulm} in 1879.\"\n"
        example_version = named_version_example if self.named_only else non_named_version_example
        examples = "Here are a few examples:\n" \
                   "Input: \"Albert was one of the greatest physicists of all time. He was born in Ulm in 1879.\"\n" \
                   + example_version + \
                   "Input: \"Germany beat Brazil 7 to 1 in the World Cup. It was a historic loss.\"\n" \
                   "Output: \"[Germany]{Germany_national_football_team} beat " \
                   "[Brazil]{Brazil_national_football_team} 7 to 1 in the [World Cup]{2014_FIFA_World_Cup}. " \
                   "It was a historic loss.\"\n"

        system_prompt = instructions + examples
        trial_num = 0
        response = None
        predictions = {}
        while trial_num < 3 and not response:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ])
            except openai.error.ServiceUnavailableError:
                print(f"Error trying to access the OpenAI API. Trying again...")
                trial_num += 1
                time.sleep(trial_num * 5)

        if response.choices:
            # Bring predictions into the correct format
            response_text = response.choices[0].message.content
            predictions = self.parse_annotated_text(text, response_text)

        return predictions

    def parse_annotated_text(self, text: str, annotated_text: str):
        """
        >>> entity_db = EntityDatabase()
        >>> entity_db.load_wikipedia_to_wikidata_db()
        >>> entity_db.load_redirects()
        >>> gpt_linker = GPTLinker(entity_db, {})
        >>> text = "Albert was one of the greatest physicists of all time."
        >>> annotated_text = "[Albert]{Albert_Einstein} was one of the greatest [physicists]{Physicist} of all time."
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((0, 6), 'Q937'), ((31, 41), 'Q169470')]
        >>> text = "Eric preferred to play Blues instead of Rock, so he joined Mayall 's Rock band."
        >>> annotated_text = "[Eric]{Eric_Clapton} preferred to play [Blues]{Blues} instead of [Rock]{Rock_music}, " \
        "so he joined [Mayall]{John_Mayall}'s [Rock]{Rock_music} band."
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((0, 4), 'Q48187'), ((23, 28), 'Q9759'), ((40, 44), 'Q11399'), ((59, 65), 'Q316282'), ((69, 73), 'Q11399')]
        >>> text = "Columbia was aquired by Sony."
        >>> annotated_text = "[Columbia]{Columbia_Records} was acquired by [Sony]{Sony_Music}."
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((0, 8), 'Q183387'), ((24, 28), 'Q56760250')]
        >>> text = "Or should I listen to the doctors the Goteborg University in Sweden?"
        >>> annotated_text = "Or should I listen to the doctors at [Goteborg University]{University_of_Gothenburg} " \
        "in [Sweden]{Sweden}?"
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((38, 57), 'Q371522'), ((61, 67), 'Q34')]
        >>> text = "What happens with a NIL prediction?"
        >>> annotated_text = "What happens with a [NIL]{} prediction?"
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((20, 23), None)]
        >>> text = "What happens with a NIL prediction?"
        >>> annotated_text = "What happens with a [NIL] prediction?"
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((20, 23), None)]
        >>> text = "Cruz, and Harper Seven."
        >>> annotated_text = "[Cruz]{Cruz_Beckham}, and [Harper_Seven]{Harper_Seven}."
        >>> predictions = gpt_linker.parse_annotated_text(text, annotated_text)
        >>> [(p.span, p.entity_id) for p in sorted(predictions.values(), key=lambda x: x.span[0])]
        [((0, 4), None)]
        """
        text_pos = 0
        ann_pos = 0
        inside_mention = False
        start = 0
        predictions = {}
        while text_pos < len(text) and ann_pos < len(annotated_text):
            text_char = text[text_pos]
            ann_char = annotated_text[ann_pos]
            if ann_char == "[" and ann_char != text_char:
                ann_pos += 1
                start = text_pos
                inside_mention = True
            elif ann_char == "]" and ann_char != text_char and inside_mention:
                end = text_pos
                if ann_pos + 1 < len(annotated_text) and not annotated_text[ann_pos + 1] == "{":
                    # Try to recover if GPT marked the entity but did not annotate it with a label.
                    logger.warning(f"No entity name label found at \"{get_text_after(ann_pos, annotated_text)} ...\""
                                   f"Predicting NIL.")
                    entity_id = None
                    ann_pos += 1
                else:
                    name_start = ann_pos + 2
                    name_end = annotated_text.find("}", name_start)
                    name = annotated_text[name_start:name_end]
                    entity_id = KnowledgeBaseMapper.get_wikidata_qid(name, self.entity_db)
                    ann_pos = name_end + 1

                span = start, end
                predictions[span] = EntityPrediction(span, entity_id, {entity_id})
            elif ann_char == text_char:
                ann_pos += 1
                text_pos += 1
            else:
                # Try to recover if predicted text deviates from original text:
                # Jump to the next prediction where predicted mention text and original text match.
                recovered = False
                first_iter = True
                while ann_pos < len(annotated_text) and text_pos < len(text) and ann_pos != -1:
                    verbose = False
                    if annotated_text[ann_pos] and not annotated_text[ann_pos].isspace():
                        verbose = True
                        logger.info("Trying to recover from deviating predicted text by jumping to the next prediction")
                        if first_iter:
                            logger.info(f"Original text:  \"{get_text_after(text_pos, text)}\"\n")
                            logger.info(f"Predicted text: \"{get_text_after(ann_pos, annotated_text)}\"")
                    first_iter = False
                    ann_pos = annotated_text.find("[", ann_pos)
                    mention_end = annotated_text.find("]", ann_pos)
                    mention_text = annotated_text[ann_pos + 1:mention_end]
                    if ann_pos == -1:
                        break
                    if mention_text:
                        if verbose:
                            logger.info(f"Next prediction is for mention \"{mention_text}\"")
                        new_text_pos = text.find(mention_text, text_pos)
                        if new_text_pos != -1:
                            recovered = True
                            text_pos = new_text_pos
                            logger.info(f"Found mention. Recovered.")
                            break
                        elif verbose:
                            logger.info("Could not find mention text in original text.")
                    ann_pos += 1

                if not recovered:
                    # Stop parsing if parser could not recover from the text deviation.
                    if not ann_pos < 0 and (text_pos < 0 or not text[text_pos:].isspace()):
                        # Otherwise, GPT simply did not add newlines or whitespaces at the end of the text.
                        # This is not a problem though, so don't print a warning otherwise.
                        logger.warning(f"\nParser error with {text_pos} "
                                       f"(\"{get_text_after(text_pos, text)} ...\"), {ann_pos} "
                                       f"(\"{get_text_after(ann_pos, annotated_text)} ...\")")
                    break
        return predictions

    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
