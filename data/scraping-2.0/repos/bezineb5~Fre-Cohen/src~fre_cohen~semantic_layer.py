""" Methods to add semantic information to the data fields.
"""

import json
import logging
from abc import ABC, abstractmethod
from re import S
from typing import Dict, Optional, Sequence

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    CompositeField,
    Edge,
    Field,
    FieldsGraph,
    IntentType,
    LinkEnum,
    RichField,
)
from fre_cohen.llms import (
    DEFAULT_RETRIES,
    LLMQualityEnum,
    build_llm_chain,
    retry_on_error,
)

logger = logging.getLogger(__name__)


class SemanticInterpretation(ABC):
    """Abstract base class for semantic interpretation"""

    def __init__(self, fields: list[Field]):
        self._fields = fields

    @abstractmethod
    def get_data_structure(self) -> FieldsGraph:
        """Returns the data structure"""


class AllSemanticInfo(BaseModel):
    """Semantic information"""

    descriptions: Dict[str, str] = PyField(
        {}, description="Dictionnary of descriptions, indexed by field name"
    )
    units: Dict[str, str] = PyField(
        {}, description="Dictionnary of units, indexed by field name"
    )


class AggregationInfo(BaseModel):
    """Aggregation information"""

    field_names: list[str] = PyField([], description="The fields to aggregate")
    name: str = PyField("", description="The name of the composite field")
    description: str = PyField(
        "", description="A rich description of the composite field"
    )


class GroupingInfo(BaseModel):
    """Grouping information"""

    composite_fields: list[AggregationInfo] = PyField(
        [], description="The composite fields"
    )


class LinkCompositeFields(BaseModel):
    """Link between data structures"""

    from_field: str = PyField(description="The data structure from")
    to_field: str = PyField(description="The data structure to")
    link: LinkEnum = PyField(LinkEnum.NONE, description="The link between the fields")


class CausalInfo(BaseModel):
    """Causal information"""

    links: list[LinkCompositeFields] = PyField(
        [], description="The links between the data structures"
    )


class IntentInfo(BaseModel):
    """Intent information"""

    intents: list[IntentType] = PyField(
        [], description="The intent of the visualization"
    )

    description: str = PyField("", description="The description of the data structure")


class OpenAISemanticInterpretation(SemanticInterpretation):
    """Semantic interpretation using OpenAI API"""

    def __init__(self, config: configuration.Config, fields: list[Field]):
        super().__init__(fields)

        self._llm_grouping = self._build_llm_chain_for_grouping(config)
        self._llm_links = self._build_llm_chain_for_links(config)
        self._llm_intent = self._build_llm_chain_for_intent(config)
        self._llm_enrich_all_fields = self._build_llm_chain_for_all_rich_field(config)

    def get_data_structure(self) -> FieldsGraph:
        """Returns the data structure"""

        # Enrich the field data
        rich_fields = self._enrich_all_fields(self._fields)

        # Group related fields
        grouped_structure = self._group_fields(rich_fields)

        def find_composite_field_index(name: str) -> int:
            return next(i for i, cf in enumerate(grouped_structure) if cf.name == name)

        # Determine links between fields
        links = self._find_links(grouped_structure)
        edges = [
            Edge(
                source=find_composite_field_index(l.from_field),
                target=find_composite_field_index(l.to_field),
                link=l.link,
            )
            for l in links
            if l.link != LinkEnum.NONE
        ]

        # Determine the intent
        intent_info = self._determine_intents(grouped_structure)

        # Build the graph
        return FieldsGraph(
            nodes=grouped_structure,
            edges=edges,
            intents=intent_info.intents,
            description=intent_info.description,
        )

    def _determine_intents(
        self, composite_fields: Sequence[CompositeField]
    ) -> IntentInfo:
        """Determines the intent of the visualization"""
        input_data = {
            "all_composite_field_names": [
                composite_field.name for composite_field in composite_fields
            ],
        }

        logger.debug("Intent LLM input: %s", input_data)
        try:
            intent_info: IntentInfo = self._llm_intent.run(input_data)
            logger.debug("Intent LLM output: %s", intent_info)
        except Exception as ex:
            logger.error("Failed to run intent LLM: %s", ex)
            intent_info = IntentInfo(intents=[], description="")

        return intent_info

    def _find_links(
        self, composite_fields: Sequence[CompositeField]
    ) -> Sequence[LinkCompositeFields]:
        """Finds the links between the data structures"""
        input_data = {
            "all_composite_field_details": "\n".join(
                [
                    f"{composite_field.name}: {composite_field.description}"
                    for composite_field in composite_fields
                ]
            ),
        }

        logger.debug("Links LLM input: %s", input_data)
        links: CausalInfo = self._llm_links.run(input_data)
        logger.debug("Links LLM output: %s", links)

        return links.links

    @retry_on_error
    def _group_fields(self, fields: Sequence[RichField]) -> Sequence[CompositeField]:
        """Group fields together"""

        retries = 0
        exception_instructions: list[str] = []
        grouping_info: Optional[GroupingInfo] = None

        while retries < DEFAULT_RETRIES:
            try:
                input_data = {
                    "all_field_names": [field.field.name for field in fields],
                    "exception_instructions": "\n".join(exception_instructions),
                }

                logger.debug("Group LLM input: %s", input_data)
                grouping_info = self._llm_grouping.run(input_data)
                logger.debug("Group LLM output: %s", grouping_info)
                if not grouping_info:
                    raise ValueError("No gouping info returned by LLM")

                grouped_ds = [
                    CompositeField(
                        columns=self._string_to_richfields(group.field_names, fields),
                        name=group.name,
                        description=group.description,
                    )
                    for group in grouping_info.composite_fields
                ]

                # Add missing fields not belonging to any group
                included_in_groups = [
                    field.field.name for group in grouped_ds for field in group.columns
                ]

                for field in fields:
                    if field.field.name not in included_in_groups:
                        grouped_ds.append(
                            CompositeField(
                                columns=[field],
                                name=field.field.name,
                                description=field.description,
                            )
                        )

                return grouped_ds
            except Exception as e:
                logger.warning("Error in Vega LLM: %s", e)
                exception_instructions = [
                    f"Please correct your previous answer: {json.dumps(grouping_info)}",
                    f"Because it contains this error: {repr(e)}",
                ]
                retries += 1
        raise RuntimeError(f"Vega LLM failed after {DEFAULT_RETRIES} retries")

    def _string_to_richfields(
        self, fields_to_lookup: Sequence[str], all_fields: Sequence[RichField]
    ) -> Sequence[RichField]:
        """Converts a list of field names to a list of RichField in the order of the field names"""
        return [
            self._lookup_field(field_name, all_fields)
            for field_name in fields_to_lookup
        ]

    def _lookup_field(
        self, field_name: str, all_fields: Sequence[RichField]
    ) -> RichField:
        """Looks up a field by name"""
        for field in all_fields:
            if field.field.name == field_name:
                return field

        raise ValueError(f"Field {field_name} not found in {all_fields}")

    def _enrich_all_fields(self, fields: Sequence[Field]) -> list[RichField]:
        """Enriches all the fields with semantic information"""
        fields_summary = "\n".join(
            [f'{field.name}: "{field.summary}"' for field in fields]
        )
        input_data = {
            "all_fields_details": fields_summary,
        }
        logger.debug("Enrich LLM input: %s", input_data)
        output: AllSemanticInfo = self._llm_enrich_all_fields.run(input_data)
        logger.debug("Enrich LLM output: %s", output)

        # Iterate over existing fields
        return [
            RichField(
                field=field,
                unit=output.units.get(field.name, ""),
                description=output.descriptions.get(field.name, ""),
            )
            for field in fields
        ]

    def _build_llm_chain_for_all_rich_field(
        self, config: configuration.Config
    ) -> LLMChain:
        """Builds a LLMChain to enrich a field"""
        return build_llm_chain(
            config,
            AllSemanticInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Here are the fields composing our data set: {all_fields_details}"
                ),
                SystemMessagePromptTemplate.from_template(
                    "Units are important to understand the data. Please write them as symbols."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Can you provide a brief description and the units of measurement for each of the fields in the structure?"
                ),
            ],
        )

    def _build_llm_chain_for_grouping(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to aggregate fields"""
        return build_llm_chain(
            config,
            GroupingInfo,
            [
                SystemMessagePromptTemplate.from_template(
                    "The order of the fields is important, consecutive fields are more closely related."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_field_names} - which fields would you group together to form meaningful mathematical entities? Please give a name to each group with the type of entity it is."
                ),
                SystemMessagePromptTemplate.from_template(
                    "A good example of composite field is a matrix made of its values. Taken individually, the values are meaningless, but together they form a matrix."
                ),
                SystemMessagePromptTemplate.from_template(
                    "Please don't link fields that are not related. Those will be always linked together, so any mistake will be propagated to the whole data structure. Example: time and position are 2 different concepts, so they should not be linked together. Only mathematical entities of the same unit can be in a composite."
                ),
                SystemMessagePromptTemplate.from_template(
                    "{exception_instructions}",
                ),
            ],
            quality=LLMQualityEnum.SPEED,
        )

    def _build_llm_chain_for_links(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to determine the links between the fields"""
        return build_llm_chain(
            config,
            CausalInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_composite_field_details} - what do you think the causality link are between those fields?"
                ),
            ],
        )

    def _build_llm_chain_for_intent(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to determine the intent of the visualization"""
        return build_llm_chain(
            config,
            IntentInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_composite_field_names} - What is the main purpose of the visualization? What insights do you hope to gain from it? How would you describe the data structure?"
                ),
            ],
        )
