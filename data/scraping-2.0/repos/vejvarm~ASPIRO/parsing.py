import logging
import os
import pathlib
import re
import json
from copy import copy

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from parent import parent as parent_metric

from enum import Enum, auto
from typing import TypeVar
from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import (
    BaseOutputParser,
    OutputParserException,
    PromptValue,
)
from flags import (CONSTANT_PLACEHOLDER, TemplateErrors, ERROR_MESSAGES, Templates, RDFExampleFormat,
                   PROMPT_TEMPLATES_FOLDER, ModelChoices, API_KEY_JSON, SUBJECT, RELATION, OBJECT,
                   OPENAI_REQUEST_TIMEOUT)
from helpers import setup_logger, make_json_compliant
from models import LLMBuilder

NAIVE_COMPLETION_RETRY_WITH_ERROR = PROMPT_TEMPLATES_FOLDER.joinpath("retry.tmp").open().read()
NAIVE_RETRY_WITH_ERROR_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY_WITH_ERROR)

T = TypeVar("T")


def prepare_prompt_and_parser(template_file: Templates, example_format: RDFExampleFormat):
    prompt_template = template_file.value.open().read()
    prompt_template = prompt_template.replace("<CONSTANT_PLACEHOLDER>", CONSTANT_PLACEHOLDER)
    prompt_template = prompt_template.replace(f"{{{example_format.value}}}", f"{{{RDFExampleFormat.DEFAULT.value}}}")

    # Prepare parser
    if "json" in template_file.value.name:
        metadata = json.load(template_file.value.with_suffix(".json").open())
        parser = JSONOutputParser.from_metadata(first_key=metadata["first_key"], output_key=metadata["output_key"],
                                                output_type=metadata["output_type"])
    else:
        parser = TextOutputParser()

    # Prepare prompt
    partial_variables = dict()
    if "{format_instructions}" in prompt_template:
        format_instructions = parser.get_format_instructions(template_file)
        partial_variables["format_instructions"] = format_instructions

    input_variables = [RDFExampleFormat.DEFAULT.value]
    if "{subjects}" in prompt_template:
        input_variables.append("subjects")
    if "{relation}" in prompt_template:
        input_variables.append("relation")
    if "{objects}" in prompt_template:
        input_variables.append("objects")

    prompt = PromptTemplate(template=prompt_template, input_variables=input_variables,
                            partial_variables=partial_variables
                            )
    return prompt, parser


def build_output_dict(output: str, error_codes: list[str], error_messages: list[str],
                      rdf_example: str = None, subj_labels: list[str] = None, obj_labels: list[str] = None,
                      shot: int = 0) -> dict:
    return {"output": output,
            "error_codes": error_codes,
            "error_messages": error_messages,
            "input_data": rdf_example if rdf_example is not None else '',
            "subjects": list(subj_labels) if subj_labels is not None else [''],
            "objects": list(obj_labels) if obj_labels is not None else [''],
            "shot": shot}


def subject_missing(string: str) -> bool:
    return bool(SUBJECT not in string)


def object_xor_value(string: str) -> bool:
    return bool((OBJECT in string and CONSTANT_PLACEHOLDER in string) or not (OBJECT in string or CONSTANT_PLACEHOLDER in string))


def misplaced_value(string: str, fetched_example_list: list) -> bool:
    return bool(CONSTANT_PLACEHOLDER in string and any(rdf['oid'] for rdf in fetched_example_list))


def contains_illegal_placeholder(string: str) -> bool:
    # Step 1: Match all <...> and <<...>> patterns
    all_patterns = re.findall(r'(<[^>]*>|<<[^>]*>>)', string)

    # Step 2: Filter out CONSTANT_PLACEHOLDER, <subject>, <object>, and <<...>> patterns
    invalid_patterns = [pattern for pattern in all_patterns if pattern not in [CONSTANT_PLACEHOLDER, '<subject>', '<object>'] and not pattern.startswith('<<')]

    return bool(invalid_patterns)


def parse_subj_obj(answer_from_llm: str, s_labels: list[str], o_labels: list[str]) -> str:
    template = copy(answer_from_llm)

    for s_label, o_label in zip(s_labels, o_labels):

        # Use regexp for replacement of subject and object
        # Create both space and underscore versions of each label pattern
        s_label_space_pattern = re.escape(s_label).replace("_", " ")
        s_label_underscore_pattern = re.escape(s_label).replace(" ", "_")

        o_label_space_pattern = re.escape(o_label).replace("_", " ")
        o_label_underscore_pattern = re.escape(o_label).replace(" ", "_")

        if SUBJECT not in template:
            template = re.sub(r'(?:\b|_)' + s_label_space_pattern + r'(?:\b|_)', SUBJECT, template, count=1,
                              flags=re.IGNORECASE)
            if SUBJECT not in template:  # If still not replaced
                template = re.sub(r'(?:\b|_)' + s_label_underscore_pattern + r'(?:\b|_)', SUBJECT, template, count=1,
                                  flags=re.IGNORECASE)

        if OBJECT not in template:
            if o_label == CONSTANT_PLACEHOLDER:
                template = re.sub(r'(?:\b|_)' + o_label_space_pattern + r'(?:\b|_)', CONSTANT_PLACEHOLDER, template,
                                  count=1, flags=re.IGNORECASE)
                if OBJECT not in template:  # If still not replaced
                    template = re.sub(r'(?:\b|_)' + o_label_underscore_pattern + r'(?:\b|_)', CONSTANT_PLACEHOLDER,
                                      template, count=1, flags=re.IGNORECASE)
            else:
                template = re.sub(r'(?:\b|_)' + o_label_space_pattern + r'(?:\b|_)', OBJECT, template, count=1,
                                  flags=re.IGNORECASE)
                if OBJECT not in template:  # If still not replaced
                    template = re.sub(r'(?:\b|_)' + o_label_underscore_pattern + r'(?:\b|_)', OBJECT, template, count=1,
                                      flags=re.IGNORECASE)

        # if regexp fails, try just simple replacement
        if SUBJECT not in template:
            template = template.replace(s_label, SUBJECT, 1)

        if OBJECT not in template:
            template = template.replace(o_label, OBJECT, 1)

    return template


def parse_relation(text: str, rel_lab: str):
    return text.replace(RELATION, rel_lab, 1)


def parse_table_format(inp_table_format_string: str):
    # Remove 'Table: ' from the string
    stripped_string = inp_table_format_string.replace('Table: ', '')

    # Split the string by '\n'
    split_by_newline = stripped_string.split('\n')

    # Get the middle part of each line and construct the result list
    result = [['<subject>', s.split(' | ')[1], '<object>'] for s in split_by_newline]

    return result


class TextOutputParser(BaseOutputParser):

    def get_format_instructions(self, template_file: Templates = "default") -> str:
        try:
            return template_file.value.with_suffix(".format_instructions").open().read()
        except FileNotFoundError:
            return Templates.DEFAULT.value.with_suffix(".format_instructions").open().read()

    def parse(self, text: str, metadata: dict = None) -> dict:
        return self._parse_text(text, metadata)

    def _parse_text(self, text: str, metadata: dict = None) -> dict:
        text = text.strip()
        text_label = "Text: "
        if text_label in text:
            text = text[text.rfind(text_label)+len(text_label):]
        if metadata is not None:
            text = parse_subj_obj(text, metadata["subj_labels"], metadata["obj_labels"])
            text = parse_relation(text, metadata["relation_label"])
        if not text.endswith("."):
            text = text+"."
        errors = []
        # SUBJECT entity errors
        if SUBJECT not in text:
            errors.append(TemplateErrors.NO_SUBJECT)
            # raise OutputParserException(TemplateErrors.NO_SUBJECT.value)
        elif text.count(SUBJECT) > 1:
            errors.append(TemplateErrors.MULTIPLE_SUBJECTS)
            # raise OutputParserException(TemplateErrors.MULTIPLE_SUBJECTS.value)
        # OBJECT entity errors:
        if CONSTANT_PLACEHOLDER == OBJECT:
            obj = CONSTANT_PLACEHOLDER
        elif CONSTANT_PLACEHOLDER in text:
            if OBJECT in text:
                errors.append(TemplateErrors.OBJECT_XOR_VALUE)
                # raise OutputParserException(TemplateErrors.OBJECT_XOR_VALUE.value)
            obj = CONSTANT_PLACEHOLDER
        else:
            obj = OBJECT
        if f"{obj}" not in text:
            errors.append(TemplateErrors.NO_OBJECT)
            # raise OutputParserException(TemplateErrors.NO_OBJECT.value.format(obj=obj))
        elif text.count(f"{obj}") > 1:
            errors.append(TemplateErrors.MULTIPLE_OBJECTS)
            # raise OutputParserException(TemplateErrors.MULTIPLE_OBJECTS.value.format(obj=obj))
        # PLACEHOLDER mismatch errors
        if contains_illegal_placeholder(text):
            errors.append(TemplateErrors.ILLEGAL_PLACEHOLDER)
            # raise OutputParserException(TemplateErrors.ILLEGAL_PLACEHOLDER.value)

        # if there are any errors, raise OutputParserException
        if metadata is None:
            rdf_example = None
            subj_labels = None
            obj_labels = None
        else:
            rdf_example = metadata["rdf_example"] if "rdf_example" in metadata.keys() else None
            subj_labels = metadata["subj_labels"] if "subj_labels" in metadata.keys() else None
            obj_labels = metadata["obj_labels"] if "obj_labels" in metadata.keys() else None

        output_message = build_output_dict(text, [], [], rdf_example, subj_labels, obj_labels)

        if errors:
            for err in errors:
                output_message["error_codes"].append(err.value)
                output_message["error_messages"].append(ERROR_MESSAGES[err])
            raise OutputParserException(json.dumps(output_message))

        return output_message

    @property
    def _type(self) -> str:
        return "structured"


class JSONOutputParser(TextOutputParser):
    first_key: str
    output_key: str
    output_type: str

    @classmethod
    def from_metadata(cls, first_key: str, output_key: str, output_type = "json") -> "JSONOutputParser":
        """

        :param first_key: [str] dict key of the first entry
        :param output_key: [str] dict key of the output entry
        :param output_type: (opt) [str] either of `text` for plain text output or `json` for json structure at the output
        :return: JSONOutputParser
        """
        cls.first_key = first_key
        cls.output_key = output_key
        cls.output_type = output_type
        return cls(first_key=first_key, output_key=output_key, output_type=output_type)

    def get_metadata(self):
        return {"first_key": self.first_key, "output_key": self.output_key}

    def parse(self, json_str: str, metadata: dict = None) -> dict:
        if self.output_type == "text":
            try:
                # first try parsing it as text (relevant for v20_json prompt)
                return self._parse_text(json_str, metadata)
            except OutputParserException:
                parsed_dict = self._parse_json(json_str)
        else:
            parsed_dict = self._parse_json(json_str)
        try:
            text = parsed_dict[self.output_key]
        except KeyError as err:
            output_message = build_output_dict(json_str, [TemplateErrors.JSON_PARSING.value],
                                               [f"KeyError: [output] must have {err} key"])
            raise OutputParserException(json.dumps(output_message))

        return self._parse_text(text, metadata)

    def _parse_json(self, json_str: str) -> dict:
        try:
            if json_str is None:
                raise json.decoder.JSONDecodeError("Expecting json format but got `None`", "", 0)
            begin_pos = json_str.find(f'"{self.first_key}":')  # start parsing from self.first_key
            if begin_pos >= 0:
                json_str = json_str[begin_pos:]
            json_str = make_json_compliant(json_str)
            parsed_dict = json.loads(json_str)
        except json.decoder.JSONDecodeError as err:
            output_message = build_output_dict(json_str, [TemplateErrors.JSON_PARSING.value], [f"Could not parse [output] as valid json ({repr(err)})"])
            raise OutputParserException(json.dumps(output_message))

        return parsed_dict


class ConsistencyValidator:
    # TODO: implement this as a customLLM
    # (https://python.langchain.com/en/latest/modules/models/llms/examples/custom_llm.html)

    class Metrics(Enum):
        PARENT = auto()

    def __init__(self, metric: Metrics, threshold: float, llm_builder: LLMBuilder, model_choice: ModelChoices, prompt_template: str,
                 source_data_key: str = "data", first_key: str = None, output_key: str = None,
                 temperature=0., max_tokens: int = 100, stop: list[str] = (".\n", "\n"),
                 logger: logging.Logger = None, path_to_jsonl_results_file: pathlib.Path = None, **model_kwargs):
        assert metric in self.Metrics
        assert 0 < threshold < 1
        assert model_choice in ModelChoices
        if path_to_jsonl_results_file is not None:
            assert path_to_jsonl_results_file.suffix == ".jsonl"

        self.metric = metric
        self.threshold = threshold
        self.model_choice = model_choice
        self.source_data_key = source_data_key
        self.first_key = first_key
        self.output_key = output_key
        self.prompt = self._prepare_prompt_template(prompt_template)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        llm = llm_builder.initialize_llm(model_choice, temperature=self.temperature,
                                         max_tokens=self.max_tokens, stop_sequences=self.stop, **model_kwargs)
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
        self.logger = logger if logger is not None else setup_logger(__name__, logging.WARNING)
        self.results_file = path_to_jsonl_results_file

    def _parse_answer(self, llm_answer: str) -> dict:
        try:
            if llm_answer is None:
                raise json.decoder.JSONDecodeError("Expecting json format but got `None`", "", 0)
            begin_pos = llm_answer.find(f'"{self.first_key}":')  # start parsing from self.first_key
            if begin_pos >= 0:
                llm_answer = llm_answer[begin_pos:]
            llm_answer = llm_answer.lstrip("{`\n").rstrip("`\n}")
            llm_answer = make_json_compliant(llm_answer)
            llm_answer = llm_answer if llm_answer.startswith("{") else "{"+llm_answer
            llm_answer = llm_answer if llm_answer.endswith("}") else llm_answer+"}"
            parsed_dict = json.loads(llm_answer)
            parsed_answer = parsed_dict[self.output_key]
        except json.decoder.JSONDecodeError:
            return llm_answer
        except KeyError:
            return json.dumps(parsed_dict)

        return parsed_answer

    @staticmethod
    def _prepare_prompt_template(template_str: str):

        assert template_str.find("{template}") > 0, "prompt template file must contain '{template}' entry"
        assert template_str.find("{data}") > 0, "prompt template file must contain '{data}' entry"

        return PromptTemplate.from_template(template_str)

    @staticmethod
    def _calc_parent(text: str, reference: str, table: list[list[str, list[str], str]]):
        _, _, f1 = parent_metric([text], [reference], [table],
                                 max_order=4,
                                 lambda_weight=1.,
                                 avg_results=True,
                                 n_jobs=1,
                                 use_tqdm=False)

        return f1


    @staticmethod
    def _make_table_map(sentence: str, rel_label: str):
        placeholders = [SUBJECT, OBJECT, CONSTANT_PLACEHOLDER]
        pattern = '|'.join(placeholders)
        sequence = re.findall(pattern, sentence)
        try:
            table_map = [[sequence[0], rel_label.split(" "), sequence[1]]]
        except IndexError:
            table_map = [[SUBJECT, rel_label.split(" "), OBJECT]]  # default
        return table_map

    # TODO: this is actually the part to replace by LLM?
    def _parse_statistic(self, text: str, **kwargs):
        reference: str = kwargs["reference"] if "reference" in kwargs.keys() else None
        relation_label: str = kwargs["relation_label"] if "relation_label" in kwargs.keys() else None

        if self.metric == self.Metrics.PARENT:
            assert reference is not None
            assert relation_label is not None
            table_map = self._make_table_map(text, relation_label)
            score: float = self._calc_parent(text, reference, table_map)
        else:
            raise NotImplementedError(f"`metric` must be one of {list(self.Metrics)} (got {self.metric})")

        if score < self.threshold:
            d = True
        else:
            d = False

        return d, score, table_map

    def _log_results(self, data_row: dict):
        with self.results_file.open("a") as f:
            f.write(json.dumps(data_row)+"\n")

    def run(self, text: str, metadata: dict, keep_better: bool):
        """

        :param text: reference text to run dehalucination on
        :param metadata: dictionary of necessary information for calculation of the statistic
        :param keep_better: (bool) if True keeps the result with higher score as the final sentence, else, returns the dehalucinated one
        :return: if keep_better == False: text which is checked and dehalucinated (if flagged by metric as hallucinated)
                 if keep_better == True: only returns dehalucinated text if the score is better than the original
        """
        assert self.source_data_key in metadata.keys()

        flag_dehalucinate, score, table_map = self._parse_statistic(text, **metadata)

        result_data = {"flag": flag_dehalucinate,
                       "threshold": self.threshold,
                       "table_map": table_map,
                       "metadata": metadata,
                       "model": self.model_choice.name,
                       "metric": self.metric.name,
                       "original": {"score": score, "text": text}}

        if flag_dehalucinate:
            llm_answer = self.chain.run(template=text, data=metadata[self.source_data_key])
            text = self._parse_answer(llm_answer)
            d_str = "T" if flag_dehalucinate else "F"
            self.logger.info(f"score:{score:.2f}({d_str}) \t text: ({result_data['original']['text']}) -> ({text}) \t table_map: {table_map}")

        _, score, _ = self._parse_statistic(text, **metadata)
        result_data["new"] = {"score": score, "text": text}

        if self.results_file is not None and flag_dehalucinate:
            self._log_results(result_data)

        if keep_better:
            if result_data["original"]["score"] > score:
                return result_data["original"]["text"]
            else:
                return text


class MultiRetryParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors.

    Does this by passing the original prompt, the completion, AND the error
    that was raised to another language and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """
    parser: BaseOutputParser[T]
    retry_chains = [LLMChain]

    @classmethod
    def from_llms(
            cls,
            llms: list[BaseLanguageModel],
            parser: BaseOutputParser[T],
            prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
    ) -> "MultiRetryParser":
        chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]
        return cls(parser=parser, retry_chains=chains)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue, shot=0, max_shots=0, metadata=None) -> T:
        if shot >= max_shots:
            return shot, self.parser.parse(completion, metadata=metadata)

        try:
            return shot, self.parser.parse(completion, metadata=metadata)
        except OutputParserException as e:
            shot += 1
            # print(f"retry attempt {shot}", end=", ")
            m = min(shot, len(self.retry_chains) - 1)
            # print(f"shot: {shot}, LLM[m]: {self.retry_chains[m].llm}")  # NOTE: DEBUG
            new_completion = self.retry_chains[m].run(
                prompt=prompt_value.to_string(), completion=completion, error=str(json.loads(str(e))["error_messages"])
                )
            return self.parse_with_prompt(new_completion, prompt_value, shot=shot, max_shots=max_shots, metadata=metadata)

    def parse(self, completion: str, metadata=None) -> T:
        return self.parser.parse(completion, metadata=metadata)

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()


if __name__ == "__main__":
    EXAMPLE_STRINGS = {
        "No error": "The <subject> is very good at parsing <object> entities.",
        TemplateErrors.NA: "<<err-n/a>>",
        TemplateErrors.API: "<<err-api>>",
        TemplateErrors.JSON_PARSING: "<<err-parsing>>",
        TemplateErrors.NO_SUBJECT: "Today, it is a sunny day in <object>.",
        TemplateErrors.MULTIPLE_SUBJECTS: "<subject> and <subject> went to the <object>.",
        TemplateErrors.NO_OBJECT: "<subject> went to the.",
        TemplateErrors.MULTIPLE_OBJECTS: "<subject> went to the <object> and <object>.",
        TemplateErrors.OBJECT_XOR_VALUE: "<subject> has a <object> and a <value>.",
        TemplateErrors.MISPLACED_VALUE: "<subject> went to the <value>.",
        TemplateErrors.ILLEGAL_PLACEHOLDER: "<subject> went to the <location>.",
        TemplateErrors.INFORMATION_LEAK: "<subject:John Doe> went to the <object:library>.",
        TemplateErrors.ERR: "<<err>>",
    }

    template_file = Templates.V10  # @param (NOTE! two-shot only works with V10 or above
    prompt_template = template_file.value.open().read()
    # replace <CONSTANT_PLACEHOLDER> with specific value
    prompt_template = prompt_template.replace("<CONSTANT_PLACEHOLDER>", CONSTANT_PLACEHOLDER)
    output_parser = TextOutputParser()
    format_instructions = output_parser.get_format_instructions(template_file)
    print(format_instructions)
    if "{format_instructions}" in prompt_template:
        partial_variables = {"format_instructions": format_instructions}
    else:
        partial_variables = dict()
    prompt = PromptTemplate(template=prompt_template, input_variables=["example_table_str"],
                            partial_variables=partial_variables
                            )

    prompt.format(examples="hi\nyou")

    for example in EXAMPLE_STRINGS.values():
        try:
            output_dict = output_parser.parse(example)
        except OutputParserException as err:
            output_dict = json.loads(str(err))

        print(output_dict)
