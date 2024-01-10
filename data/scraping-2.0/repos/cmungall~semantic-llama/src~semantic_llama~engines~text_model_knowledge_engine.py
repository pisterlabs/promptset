"""
Main Knowledge Extractor class.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pydantic
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from oaklib import get_implementation_from_shorthand
from oaklib.datamodels.text_annotator import TextAnnotation, TextAnnotationConfiguration

from semantic_llama.engines.engine import EXAMPLE, FIELD, OBJECT, KnowledgeEngine, ANNOTATION_KEY_PROMPT_SKIP, \
    ANNOTATION_KEY_PROMPT, ANNOTATION_KEY_ANNOTATORS
from semantic_llama.templates.core import ExtractionResult, NamedEntity

this_path = Path(__file__).parent


RESPONSE_ATOM = Union[str, "ResponseAtom"]
RESPONSE_DICT = Dict[FIELD, Union[RESPONSE_ATOM, List[RESPONSE_ATOM]]]



@dataclass
class TextModelKnowledgeEngine(KnowledgeEngine):
    """Knowledge extractor."""

    engine: str = "text-davinci-003"
    recurse: bool = True

    def extract_from_text(
        self, text: str, cls: ClassDefinition = None, object: OBJECT = None
    ) -> ExtractionResult:
        """
        Extract annotations from the given text.

        :param text:
        :param cls:
        :param object: optional stub object
        :return:
        """
        raw_text = self._raw_extract(text, cls, object=object)
        logging.info(f"RAW TEXT: {raw_text}")
        r = self.parse_completion_payload(raw_text, cls, object=object)
        return ExtractionResult(
            input_text=text,
            raw_completion_output=raw_text,
            prompt=self.last_prompt,
            results=r,
            named_entities=self.named_entities,
        )

    def _extract_from_text_to_dict(self, text: str, cls: ClassDefinition = None) -> RESPONSE_DICT:
        raw_text = self._raw_extract(text, cls)
        return self._parse_response_to_dict(raw_text, cls)

    def generalize(
        self, object: Union[pydantic.BaseModel, dict], examples: List[EXAMPLE]
    ) -> ExtractionResult:
        """
        Generalize the given examples.

        :param object:
        :param examples:
        :return:
        """
        cls = self.template_class
        sv = self.schemaview
        prompt = "example:\n"
        for example in examples:
            prompt += f"{self.serialize_object(example)}\n\n"
        prompt += "\n\n===\n\n"
        if isinstance(object, pydantic.BaseModel):
            object = object.dict()
        for k, v in object.items():
            if v:
                slot = sv.induced_slot(k, cls.name)
                prompt += f"{k}: {self._serialize_value(v, slot)}\n"
        print(f"PROMPT: {prompt}")
        payload = self.client.complete(prompt)
        prediction = self.parse_completion_payload(payload, object=object)
        return ExtractionResult(
            input_text=prompt,
            raw_completion_output=payload,
            # prompt=self.last_prompt,
            results=[prediction],
            named_entities=self.named_entities,
        )

    def map_terms(self, terms: List[str], ontology: str) -> Dict[str, List[str]]:
        """
        Map the given terms to the given ontology.

        EXPERIMENTAL

        currently GPT-3 does not do so well with this task.

        :param terms:
        :param ontology:
        :return:
        """
        # TODO: make a separate config
        examples = {
            "go": {
                "nucleui": "nucleus",
                "mitochondrial": "mitochondrion",
                "signaling": "signaling pathway",
                "cysteine biosynthesis": "cysteine biosynthetic process",
                "alcohol dehydrogenase": "alcohol dehydrogenase activity",
            },
            "uberon": {
                "feet": "pes",
                "forelimb, left": "left forelimb",
                "hippocampus": "Ammons horn",
            },
        }
        ontology = ontology.lower()
        if ontology in examples:
            example = examples[ontology]
        else:
            example = examples["uberon"]
        prompt = "Normalize the following semicolon separated list of terms to the {ontology.upper()} ontology\n\n"
        prompt += "For example:\n\n"
        for k, v in example.items():
            prompt += f"{k}: {v}\n"
        prompt += "===\n\nTerms:"
        prompt += "; ".join(terms)
        prompt += "===\n\n"
        payload = self.client.complete(prompt)
        # outer parse
        best_results = []
        for sep in ["\n", "; "]:
            results = payload.split(sep)
            if len(results) > len(best_results):
                best_results = results

        def normalize(s: str) -> str:
            s = s.strip()
            s.replace("_", " ")
            return s.lower()

        mappings = {}
        for result in best_results:
            if ":" not in result:
                logging.error(f"Count not parse result: {result}")
                continue
            k, v = result.strip().split(":", 1)
            k = k.strip()
            v = v.strip()
            for t in terms:
                if normalize(t) == normalize(k):
                    mappings[t] = v
                    break
        for t in terms:
            if t not in mappings:
                logging.warning(f"Could not map term: {t}")
        return mappings

    def serialize_object(self, example: EXAMPLE, cls: ClassDefinition = None) -> str:
        if cls is None:
            cls = self.template_class
        if isinstance(example, str):
            return example
        if isinstance(example, pydantic.BaseModel):
            example = example.dict()
        lines = []
        sv = self.schemaview
        for k, v in example.items():
            if not v:
                continue
            slot = sv.induced_slot(k, cls.name)
            v_serialized = self._serialize_value(v, slot)
            lines.append(f"{k}: {v_serialized}")
        return "\n".join(lines)

    def _serialize_value(self, val: Any, slot: SlotDefinition) -> str:
        if val is None:
            return ""
        if isinstance(val, list):
            return "; ".join([self._serialize_value(v, slot) for v in val if v])
        if isinstance(val, dict):
            return " - ".join([self._serialize_value(v, slot) for v in val.values() if v])
        sv = self.schemaview
        if slot.range in sv.all_classes():
            if self.labelers:
                labelers = list(self.labelers)
            else:
                labelers = []
            labelers += self.get_annotators(sv.get_class(slot.range))
            if labelers:
                for labeler in labelers:
                    label = labeler.label(val)
                    if label:
                        return label
        return val

    def _raw_extract(self, text, cls: ClassDefinition = None, object: OBJECT = None) -> str:
        """
        Extract annotations from the given text.

        :param text:
        :return:
        """
        prompt = self.get_completion_prompt(cls, text, object=object)
        self.last_prompt = prompt
        full_text = prompt
        payload = self.client.complete(prompt)
        return payload

    def get_completion_prompt(
        self, cls: ClassDefinition = None, text: str = None, object: OBJECT = None
    ) -> str:
        """Get the prompt for the given template."""
        if cls is None:
            cls = self.template_class
        if not text or ("\n" in text or len(text) > 60):
            prompt = (
                "From the text below, extract the following entities in the following format:\n\n"
            )
        else:
            prompt = "Split the following piece of text into fields in the following format:\n\n"
        for slot in self.schemaview.class_induced_slots(cls.name):
            if ANNOTATION_KEY_PROMPT_SKIP in slot.annotations:
                continue
            if ANNOTATION_KEY_PROMPT in slot.annotations:
                slot_prompt = slot.annotations[ANNOTATION_KEY_PROMPT].value
            elif slot.description:
                slot_prompt = slot.description
            else:
                if slot.multivalued:
                    slot_prompt = f"semicolon-separated list of {slot.name}s"
                else:
                    slot_prompt = f"the value for {slot.name}"
            prompt += f"{slot.name}: <{slot_prompt}>\n"
        # prompt += "Do not answer if you don't know\n\n"
        prompt = f"{prompt}\n\nText:\n{text}\n\n===\n\n"
        if object:
            if cls is None:
                cls = self.template_class
            if isinstance(object, pydantic.BaseModel):
                object = object.dict()
            for k, v in object.items():
                if v:
                    slot = self.schemaview.induced_slot(k, cls.name)
                    prompt += f"{k}: {self._serialize_value(v, slot)}\n"
        return prompt

    def _parse_response_to_dict(
        self, results: str, cls: ClassDefinition = None
    ) -> Optional[RESPONSE_DICT]:
        """
        Parses the pseudo-YAML response from OpenAI into a dictionary object.

        E.g.

            foo: a; b; c

        becomes

            {"foo": ["a", "b", "c"]}

        :param results:
        :return:
        """
        lines = results.splitlines()
        ann = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                logging.warning(f"Line {line} does not contain a colon")
                return
            r = self._parse_line_to_dict(line, cls)
            if r is not None:
                field, val = r
                ann[field] = val
        return ann

    def _parse_line_to_dict(
        self, line: str, cls: ClassDefinition = None
    ) -> Optional[Tuple[FIELD, RESPONSE_ATOM]]:
        if cls is None:
            cls = self.template_class
        sv = self.schemaview
        # each line is a key-value pair
        logging.info(f"PARSING LINE: {line}")
        field, val = line.split(":", 1)
        if not val:
            logging.warning(f"Cannot parse {line}")
            return
        # The LLML may mutate the output format somewhat,
        # randomly pluralizing or replacing spaces with underscores
        field = field.lower().replace(" ", "_")
        cls_slots = sv.class_slots(cls.name)
        slot = None
        if field in cls_slots:
            slot = sv.induced_slot(field, cls.name)
        else:
            if field.endswith("s"):
                field = field[:-1]
            if field in cls_slots:
                slot = sv.induced_slot(field, cls.name)
        if not slot:
            logging.error(f"Cannot find slot for {field} in {line}")
            # raise ValueError(f"Cannot find slot for {field} in {line}")
            return
        inlined = slot.inlined
        slot_range = sv.get_class(slot.range)
        if not inlined:
            if slot.range in sv.all_classes():
                inlined = sv.get_identifier_slot(slot_range.name) is None
        val = val.strip()
        if slot.multivalued:
            vals = [v.strip() for v in val.split(";")]
        else:
            vals = [val]
        vals = [val for val in vals if val]
        logging.debug(f"SLOT: {slot.name} INL: {inlined} VALS: {vals}")
        if inlined:
            transformed = False
            slots_of_range = sv.class_slots(slot_range.name)
            if self.recurse or len(slots_of_range) > 2:
                vals = [self._extract_from_text_to_dict(v, slot_range) for v in vals]
            else:
                for sep in [" - ", ":", "/", "*", "-"]:
                    if all([sep in v for v in vals]):
                        vals = [dict(zip(slots_of_range, v.split(sep, 1))) for v in vals]
                        for v in vals:
                            for k in v.keys():
                                v[k] = v[k].strip()
                        transformed = True
                        break
                if not transformed:
                    logging.warning(f"Did not find separator in {vals} for line {line}")
                    return
        # transform back from list to single value if not multivalued
        if slot.multivalued:
            final_val = vals
        else:
            if len(vals) != 1:
                logging.error(f"Expected 1 value for {slot.name} in '{line}' but got {vals}")
            final_val = vals[0]
        return field, final_val

    def parse_completion_payload(
        self, results: str, cls: ClassDefinition = None, object: dict = None
    ) -> pydantic.BaseModel:
        """
        Parse the completion payload into a pydantic class.

        :param results:
        :param cls:
        :param object: stub object
        :return:
        """
        raw = self._parse_response_to_dict(results, cls)
        print(f"RAW: {raw}")
        if object:
            raw = {**object, **raw}
        return self.ground_annotation_object(raw, cls)

    def ground_annotation_object(
        self, ann: RESPONSE_DICT, cls: ClassDefinition = None
    ) -> pydantic.BaseModel:
        """Ground the raw parse of the OpenAI payload.

        :param ann: Raw annotation object
        :param cls: schema class the ground object should instantiate
        :return: Grounded annotation object
        """
        logging.debug(f"Grounding annotation object {ann}")
        if cls is None:
            cls = self.template_class
        sv = self.schemaview
        new_ann = {}
        if ann is None:
            logging.error(f"Cannot ground None annotation, cls={cls.name}")
            return
        for field, vals in ann.items():
            if isinstance(vals, list):
                multivalued = True
            else:
                multivalued = False
                vals = [vals]
            slot = sv.induced_slot(field, cls.name)
            rng = sv.get_class(slot.range)
            new_ann[field] = []
            for val in vals:
                if not val:
                    continue
                if isinstance(val, tuple):
                    sub_slots = sv.class_induced_slots(rng.name)
                    obj = {}
                    for i in range(0, len(val)):
                        sub_slot = sub_slots[i]
                        sub_rng = sv.get_class(sub_slot.range)
                        if not sub_rng:
                            logging.error(f"Cannot find range for {sub_slot.name}")
                        result = self.ground_text_to_id(val[i], sub_rng)
                        obj[sub_slot.name] = result
                elif isinstance(val, dict):
                    obj = self.ground_annotation_object(val, rng)
                else:
                    obj = self.ground_text_to_id(val, rng)
                if multivalued:
                    new_ann[field].append(obj)
                else:
                    new_ann[field] = obj
        logging.debug(f"Creating object from dict {new_ann}")
        print(new_ann)
        py_cls = self.template_module.__dict__[cls.name]
        return py_cls(**new_ann)

    def ground_text_to_id(self, text: str, cls: ClassDefinition = None) -> str:
        """
        Ground the text to an ID.

        :param text: Text to ground, e.g. a gene symbol
        :param cls: schema class the ground object should instantiate
        :return: ID or input string if cannot be grounded
        """
        result = self.ground_text(text, cls)
        if result:
            if self.named_entities is None:
                self.named_entities = []
            self.named_entities.append(NamedEntity(id=result.object_id, label=text))
            return result.object_id
        if self.recurse and cls:
            print(f"RECURSING: {text} cls={cls.name}")
            obj = self.extract_from_text(text, cls).results
            if obj:
                if self.named_entities is None:
                    self.named_entities = []
                try:
                    obj.id = text
                except ValueError as e:
                    logging.error(f"No id for {obj} {e}")
                self.named_entities.append(obj)
        return text

    def ground_text(self, text: str, cls: ClassDefinition = None) -> Optional[TextAnnotation]:
        """
        Ground the given text to an object ID.

        :param text: text to ground, e.g. gene symbol
        :param cls: schema class the ground object should instantiate
        :return:
        """
        if cls is None:
            logging.error("No range")
            return
        logging.info(f"GROUNDING {text} using {cls.name}")
        config = TextAnnotationConfiguration(matches_whole_text=True)
        if self.annotators and cls.name in self.annotators:
            annotators = self.annotators[cls.name]
        else:
            if ANNOTATION_KEY_ANNOTATORS not in cls.annotations:
                logging.error(f"ground_text: No annotators for {cls.name}")
                return
            annotators = cls.annotations[ANNOTATION_KEY_ANNOTATORS].value.split(", ")
        logging.info(f" Annotators: {annotators}")
        for annotator in annotators:
            if isinstance(annotator, str):
                logging.info(f"Loading annotator {annotator}")
                annotator = get_implementation_from_shorthand(annotator)
            try:
                results = annotator.annotate_text(text, config)
                for result in results:
                    return result
            except Exception as e:
                logging.error(f"Error with {annotator} for {text}: {e}")
