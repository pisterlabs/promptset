"""
The cartography layer is responsible for generating map layout
"""

import json
import logging
import pathlib
from typing import Any, Dict, Optional, Sequence

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    FieldsGraph,
    GraphSpecifications,
    IndividualGraph,
    InstructionOrigin,
    RichField,
)
from fre_cohen.llms import LLMQualityEnum, build_llm_chain
from fre_cohen.visualization_layer import IndividualVisualizationLayer

logger = logging.getLogger(__name__)

MAPBOX_STYLE_FILE = (
    pathlib.Path(__file__).parent
    / "assets"
    / "klokantech-basic-gl-style-master"
    / "style.json"
)
COLOR_ATTRIBUTES = ["fill-color", "line-color", "text-color", "icon-color"]


class MapStylingSpecifications(BaseModel):
    """Styling specifications of the map"""

    map_style = PyField(
        "",
        description="Map style",
    )


class MapboxStyleSpecification(BaseModel):
    """Mapbox style specification"""

    specifications: Any = PyField(
        None,
        description="Specifications of the map style, e.g. the mapbox style specification.",
    )


class MapboxStyleLayerInfo(BaseModel):
    """Mapbox style layer info"""

    name: str = PyField(
        None,
        description="Name of the map layer",
    )
    color: str = PyField(
        None,
        description="Color of the map layer",
    )


class MapboxStyleLayers(BaseModel):
    """Mapbox style layers"""

    layers: Sequence[MapboxStyleLayerInfo] = PyField(
        None,
        description="Map layers",
    )


class LLMMapboxStyleSpecifications(IndividualVisualizationLayer):
    """Generates mapbox style specifications for the map using LLMs"""

    def __init__(
        self,
        config: configuration.Config,
        fields_graph: FieldsGraph,
        graph: IndividualGraph,
        previous_specifications: Optional[GraphSpecifications],
    ):
        super().__init__(fields_graph, graph, previous_specifications)

        self._llm_style = self._build_llm_chain_for_mapbox_style(config)

    def get_specifications(self) -> GraphSpecifications:
        """Returns the specifications of the graph"""

        # Run the LLM chain
        required_layers = self._get_mapbox_style_specifications()
        map_style = self._generate_mapbox_style_file(
            MAPBOX_STYLE_FILE, required_layers.layers
        )

        return GraphSpecifications(
            format_type="mabpox-style",
            specifications=map_style,
            graph=self._graph,
        )

    def _summarize_fields(self, fields: Sequence[RichField]) -> str:
        """Summarizes the fields"""
        return "\n".join(
            [
                f'"{field.field.name}": "{field.description}" with unit {field.unit} and with summary: "{field.field.summary}"'
                for field in fields
            ]
        )

    def _summarize_composite_fields(self, variable_indexes: Sequence[int]) -> str:
        """Summarizes the variables"""
        variables = [self._fields_graph.nodes[index] for index in variable_indexes]
        return "\n".join(
            [
                f'"{composite_field.description}" with fields:\n{self._summarize_fields(composite_field.columns)}\n'
                for composite_field in variables
            ]
        )

    def _get_mapbox_style_specifications(self) -> MapboxStyleLayers:
        """Returns the Mapbox style specifications"""

        # Build the input data
        input_data = {
            "title": self._graph.title,
            "independent_variables_summary": self._summarize_composite_fields(
                self._graph.independent_variables
            ),
            "dependent_variables_summary": self._summarize_composite_fields(
                self._graph.dependent_variables
            ),
            "map_layers": self._summarize_map_layers(),
            "critic_advices": "\n".join(
                [
                    instruction.instruction
                    for instruction in self._graph.instructions
                    if instruction.origin == InstructionOrigin.CRITIC
                ]
            ),
            "human_instructions": "\n".join(
                [
                    instruction.instruction
                    for instruction in self._graph.instructions
                    if instruction.origin == InstructionOrigin.HUMAN
                ]
            ),
        }
        logger.debug("Mapbox style LLM input: %s", input_data)
        output: MapboxStyleLayers = self._llm_style.run(input_data)
        logger.debug("Mapbox style LLM output: %s", output)
        return output

    def _summarize_map_layers(self) -> str:
        layers = self._list_style_layers_from_mapbox_style_file(MAPBOX_STYLE_FILE)
        return self._format_map_layers_for_llm(layers)

    def _format_map_layers_for_llm(self, layers: Sequence[MapboxStyleLayerInfo]) -> str:
        """Formats the map layers for LLM input"""
        return "\n".join(
            [f'"{layer.name}" with color "{layer.color}"' for layer in layers]
        )

    def _get_layer_color(self, layer: Dict[str, Any]) -> str:
        """Returns the color of the layer.
        Colors can be the line, fill, text... color depending on the type of the layer.
        """
        paint = layer.get("paint", {})
        for color_attribute in COLOR_ATTRIBUTES:
            if color_attribute in paint:
                return paint[color_attribute]
        return ""

    def _list_style_layers_from_mapbox_style_file(
        self, style_file: pathlib.Path
    ) -> Sequence[MapboxStyleLayerInfo]:
        """Lists the layers from the mapbox style file"""
        with open(style_file, encoding="utf-8") as f:
            style = json.load(f)
        return [
            MapboxStyleLayerInfo(name=layer["id"], color=self._get_layer_color(layer))
            for layer in style["layers"]
        ]

    def _generate_mapbox_style_file(
        self, style_file: pathlib.Path, layers: Sequence[MapboxStyleLayerInfo]
    ) -> str:
        """Generates the mapbox style file"""
        with open(style_file, encoding="utf-8") as f:
            style = json.load(f)

        for layer in style["layers"]:
            found = False
            # Find the layer in the list of layers
            for layer_info in layers:
                if layer["id"] == layer_info.name:
                    found = True
                    # Update the color of the layer. It can be in various attributes depending on the type of the layer.
                    for color_attribute in COLOR_ATTRIBUTES:
                        if color_attribute in layer["paint"]:
                            layer["paint"][color_attribute] = layer_info.color
                    break

            if not found:
                # Remove the layer if it is not in the list of layers
                style["layers"].remove(layer)

        return json.dumps(style)

    def _build_llm_chain_for_mapbox_style(
        self, config: configuration.Config
    ) -> LLMChain:
        """Builds the LLM chain for vega-lite specification"""
        llm_chain = build_llm_chain(
            config,
            MapboxStyleLayers,
            [
                SystemMessagePromptTemplate.from_template(
                    "This is the title of the map: {title}"
                ),
                SystemMessagePromptTemplate.from_template(
                    "These are the independent variables of the map:\n{independent_variables_summary}"
                ),
                SystemMessagePromptTemplate.from_template(
                    "These are the dependent variables of the map:\n{dependent_variables_summary}"
                ),
                SystemMessagePromptTemplate.from_template(
                    "These are the map layers:\n{map_layers}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "You should only include layers which are relevant to the data we want to visualize. Can you remove any layers which are not relevant?"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Can you adjust the colors to highlight the most important information while keeping the map readable?"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Given a set of data and a description of the map you want to create, can you list the layers with colors in JSON format that will produce the desired visualization?"
                ),
            ],
            LLMQualityEnum.SPEED,
        )

        return llm_chain
