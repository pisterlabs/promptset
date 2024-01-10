"""
Main Knowledge Extractor class.
"""
import importlib
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple, Union

import openai
import pydantic
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from oaklib import BasicOntologyInterface, get_implementation_from_shorthand
from oaklib.interfaces import TextAnnotatorInterface
from oaklib.utilities.apikey_manager import get_apikey_value

from semantic_llama.clients import OpenAIClient
from semantic_llama.templates.core import ExtractionResult, NamedEntity

this_path = Path(__file__).parent


OBJECT = Union[str, pydantic.BaseModel, dict]
EXAMPLE = OBJECT
FIELD = str
TEMPLATE_NAME = str

# annotation metamodel
ANNOTATION_KEY_PROMPT = "prompt"
ANNOTATION_KEY_PROMPT_SKIP = "prompt.skip"
ANNOTATION_KEY_ANNOTATORS = "annotators"

# TODO: introspect
DATAMODELS = [
    "treatment.DiseaseTreatmentSummary",
    "gocam.GoCamAnnotations",
    "bioloigical_process.BiologicalProcess",
    "environmental_sample.Study",
    "mendelian_disease.MendelianDisease",
    "reaction.Reaction",
    "recipe.Recipe",
]


@dataclass
class KnowledgeEngine(ABC):
    """
    Abstract base class for all engines
    """

    template: TEMPLATE_NAME
    """LinkML Template to use for this engine.
    Must be of the form <module_name>.<ClassName>"""

    template_class: ClassDefinition = None
    """LinkML Class for the template.
    This is derived from the template and does not need to be set manually."""

    template_pyclass = None
    """Python class for the template.
    This is derived from the template and does not need to be set manually."""

    template_module: ModuleType = None
    """Python module for the template.
    This is derived from the template and does not need to be set manually."""

    schemaview: SchemaView = None
    """LinkML SchemaView over the template.
    This is derived from the template and does not need to be set manually."""

    api_key: str = None
    """OpenAI API key."""

    engine: str = None
    """OpenAI Engine. This is set for each base class"""

    annotator: TextAnnotatorInterface = None
    """Default annotator. TODO: deprecate?"""

    annotators: Dict[str, List[TextAnnotatorInterface]] = None
    """Annotators for each class.
    An annotator will ground/map labels to CURIEs.
    These override the annotators annotated in the template
    """

    labelers: List[BasicOntologyInterface] = None
    """Labelers that map CURIEs to labels"""

    client: OpenAIClient = None
    """All calls to LLMs are delegated through this client"""

    named_entities: List[NamedEntity] = None
    """Cache of all named entities"""

    last_text: str = None
    """Cache of last text."""

    last_prompt: str = None
    """Cache of last prompt used."""

    def __post_init__(self):
        self.template_class = self._get_template_class(self.template)
        self.client = OpenAIClient()
        self.api_key = self._get_openai_api_key()
        openai.api_key = self.api_key

    def extract_from_text(
        self, text: str, cls: ClassDefinition = None, object: OBJECT = None
    ) -> ExtractionResult:
        raise NotImplementedError

    def extract_from_file(self, file: Union[str, Path, TextIO]) -> pydantic.BaseModel:
        """
        Extract annotations from the given text.

        :param file:
        :return:
        """
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            with file.open() as f:
                text = f.read()
        else:
            text = file.read()
        self.last_text = text
        r = self.extract_from_text(text)
        r.input_id = str(file)
        return r

    def synthesize(self, cls: ClassDefinition = None, object: OBJECT = None) -> ExtractionResult:
        pass

    def generalize(
        self, object: Union[pydantic.BaseModel, dict], examples: List[EXAMPLE]
    ) -> ExtractionResult:
        raise NotImplementedError

    def map_terms(self, terms: List[str], ontology: str) -> Dict[str, List[str]]:
        raise NotImplementedError

    def _get_template_class(self, template: TEMPLATE_NAME) -> ClassDefinition:
        """
        Get the LinkML class for a template.

        :param template: template name of the form module.ClassName
        :return: LinkML class definition
        """
        logging.info(f"Loading schema for {template}")
        module_name, class_name = template.split(".", 1)
        templates_path = this_path.parent / "templates"
        path_to_template = str(templates_path / f"{module_name}.yaml")
        mod = importlib.import_module(f"semantic_llama.templates.{module_name}")
        self.template_module = mod
        self.template_pyclass = mod.__dict__[class_name]
        sv = SchemaView(path_to_template)
        self.schemaview = sv
        logging.info(f"Getting class for template {template}")
        cls = None
        for c in sv.all_classes().values():
            if c.name == class_name:
                cls = c
                break
        if not cls:
            raise ValueError(f"Template {template} not found")
        return cls

    def _get_openai_api_key(self):
        """Get the OpenAI API key from the environment."""
        # return os.environ.get("OPENAI_API_KEY")
        return get_apikey_value("openai")

    def get_annotators(self, cls: ClassDefinition = None) -> List[BasicOntologyInterface]:
        """
        Get the annotators/labelers for a class.

        The annotators are returned in order of precedence

        Annotators are used to *ground* labels as CURIEs.
        Annotators may also do double-duty as labelers (i.e. map CURIEs to labels)

        These are specified by linkml annotations within the template/schema;
        if the engine has a set of annotators specified these take precedence.

        :param cls: schema class
        :return: list of annotations
        """
        if self.annotators and cls.name in self.annotators:
            annotators = self.annotators[cls.name]
        else:
            if ANNOTATION_KEY_ANNOTATORS not in cls.annotations:
                logging.error(f"No annotators for {cls.name}")
                return []
            annotators = cls.annotations[ANNOTATION_KEY_ANNOTATORS].value.split(", ")
        logging.info(f" Annotators: {annotators}")
        objs = []
        for annotator in annotators:
            if isinstance(annotator, str):
                logging.info(f"Loading annotator {annotator}")
                objs.append(get_implementation_from_shorthand(annotator))
            elif isinstance(annotator, BasicOntologyInterface):
                objs.append(annotator)
            else:
                raise ValueError(f"Unknown annotator type {annotator}")
        return objs

    def ground_text_to_id(self, text: str, cls: ClassDefinition = None) -> str:
        raise NotImplementedError

    def cached_completions(
        self, search_term: str = None, engine: str = None
    ) -> Iterator[Tuple[str, str]]:
        if engine is None:
            engine = self.engine
        cur = self.client.db_connection()
        res = cur.execute("SELECT prompt, payload FROM cache WHERE engine=?", (engine,))
        for row in res:
            if search_term and search_term not in row[0]:
                continue
            yield row

    def merge_resultsets(
        self, resultset: List[ExtractionResult], unique_fields: List[str] = None
    ) -> ExtractionResult:
        """
        Merges all resultsets into a single resultset.

        Note the first element of the list is mutated.

        :param resultset:
        :return:
        """
        result = resultset[0].results
        for next_extraction in resultset[1:]:
            next_result = next_extraction.results
            if unique_fields:
                for k in unique_fields:
                    if k in result and k in next_result:
                        if result[k] != next_result[k]:
                            logging.error(
                                f"Cannot merge unique fields: {k} {result[k]} != {next_result[k]}"
                            )
                            continue
            for k, v in vars(next_result).items():
                curr_v = getattr(result, k, None)
                if isinstance(v, list):
                    if all(isinstance(x, str) for x in v):
                        setattr(result, k, list(set(curr_v).union(set(v))))
                    else:
                        setattr(result, k, curr_v + v)
                else:
                    if curr_v and v and curr_v != v:
                        logging.error(f"Cannot merge {curr_v} and {v}")
                    if v:
                        setattr(result, k, v)
        return resultset[0]
