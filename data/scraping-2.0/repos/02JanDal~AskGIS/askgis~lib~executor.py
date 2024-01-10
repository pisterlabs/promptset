from dataclasses import dataclass, fields
from typing import List, Optional, Union, get_args, get_origin

from langchain.python import PythonREPL
from PyQt5.QtCore import QVariant
from qgis import processing
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingFeedback,
    QgsProject,
    QgsUnitTypes,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)

from askgis import LOGGER
from askgis.lib.util import to_snake_case


class Layer:
    pass


@dataclass
class SourceLayer(Layer):
    """id must be one of: {layer_names}"""

    id: str


@dataclass
class FilteredLayer(Layer):
    source: Layer
    field: str
    value: Union[str, float, int, bool]


@dataclass
class BufferedLayer(Layer):
    source: Layer
    distance: float


@dataclass
class BinaryOperationLayer(Layer):
    source_a: Layer
    source_b: Layer


class UnionLayer(BinaryOperationLayer):
    def __init__(self, source, *sources):
        if len(sources) > 1:
            sources = UnionLayer(*sources)
        else:
            sources = sources[0]
        super().__init__(source, sources)


class IntersectionLayer(BinaryOperationLayer):
    pass


class DifferenceLayer(BinaryOperationLayer):
    pass


class Action:
    pass


@dataclass
class SelectAction(Action):
    """Use this if the user asks for data to be selected or where something is located."""

    layer: Layer


@dataclass
class AddToMapAction(Action):
    layer: Layer


@dataclass
class CountAction(Action):
    layer: Layer


layer_functions = dict(
    get_layer=SourceLayer,
    filter=FilteredLayer,
    buffer=BufferedLayer,
    union=UnionLayer,
    intersection=IntersectionLayer,
    difference=DifferenceLayer,
)
action_functions = dict(
    select=SelectAction, add_to_map=AddToMapAction, count=CountAction
)
functions = dict(**layer_functions, **action_functions)


def get_prompt_functions() -> List[str]:
    """Get a list of available function signatures, to be used in a prompt."""

    def type_def(type) -> str:
        if get_origin(type) is Union:
            return f"Union[{', '.join(type_def(t) for t in get_args(type))}]"
        else:  # if hasattr(type, "__name__"):
            return type.__name__

    def function_def(name: str, function) -> str:
        parameters = ", ".join(
            f"{f.name}: {type_def(f.type)}" for f in fields(function)
        )
        if function.__doc__:
            doc = f'\n    """{function.__doc__}"""'
        else:
            doc = None
        return f'def {name}({parameters}) -> {"Layer" if issubclass(function, Layer) else "Action"}:{"" if doc is None else doc}\n    pass'

    return [function_def(k, v) for k, v in functions.items()]


def to_action(python: str) -> Optional[Action]:
    """Execute the Python code to collect a list of actions and their layer "trees"."""

    actions: List[Action] = []

    def make_action_func(type):
        def func(*args, **kwargs):
            actions.append(type(*args, **kwargs))

        return func

    funcs = dict(
        **layer_functions,
        **{k: make_action_func(v) for k, v in action_functions.items()},
    )

    out = PythonREPL(_globals=funcs, _locals=funcs).run(python.lstrip())
    if out.strip():
        LOGGER.warning(f"Error while executing Python for actions: {out}")

    return actions[0] if len(actions) > 0 else None


@dataclass
class VectorData:
    original: QgsVectorLayer
    """The original layer from which this data is derived. Useful to perform selection."""
    data: QgsVectorLayer
    """The data, might be the original layer or some derived layer (buffered, etc.)."""


class Executor:
    """Executes actions on the data available in the given project."""

    def __init__(self, project: QgsProject, feedback: QgsProcessingFeedback):
        self._project = project
        self._feedback = feedback

    def execute(self, action: Action) -> str:
        """Execute a single action."""

        return self._execute_action(action)

    def _execute_layer(self, layer: Layer) -> VectorData:
        return getattr(self, f"_execute_{to_snake_case(layer.__class__.__name__)}")(
            layer
        )

    def _execute_source_layer(self, layer: SourceLayer) -> VectorData:
        original = next(
            (
                l
                for l in self._project.mapLayers().values()
                if l.id() == layer.id
                or l.name().lower() == layer.id.lower()
                or l.shortName().lower() == layer.id.lower()
            ),
            None,
        )
        if not original:
            raise FileNotFoundError(f"Unknown layer: {layer.id}")
        LOGGER.warning(f"Source layer {layer.id} has {original.featureCount()} items")
        return VectorData(original=original, data=original)

    def _execute_filtered_layer(self, layer: FilteredLayer) -> VectorData:
        source = self._execute_layer(layer.source)

        field_idx = source.data.fields().lookupField(layer.field)
        if field_idx < 0:
            raise KeyError(f"Unknown field: {layer.field}")
        field_type = source.data.fields().field(field_idx).type()
        field_type_is_int = field_type in (
            QVariant.Int,
            QVariant.UInt,
            QVariant.LongLong,
            QVariant.ULongLong,
        )
        value = layer.value
        # some type coercions
        if (
            field_type_is_int
            and isinstance(value, str)
            and value.lower() in ("true", "yes")
        ):
            value = 1
        elif (
            field_type_is_int
            and isinstance(value, str)
            and value.lower() in ("false", "no")
        ):
            value = 0
        elif field_type_is_int and isinstance(value, bool):
            value = 1 if value else 0

        result = self._run_processing(
            "native:extractbyattribute",
            dict(
                INPUT=source.data,
                FIELD=layer.field,
                OPERATOR=0,  # equals
                VALUE=f"{value}",
                OUTPUT="memory:",
            ),
        )
        LOGGER.warning(
            f"Filter went from {source.data.featureCount()} to {result['OUTPUT'].featureCount()} features"
        )
        return VectorData(original=source.original, data=result["OUTPUT"])

    def _execute_buffered_layer(self, layer: BufferedLayer) -> VectorData:
        source = self._execute_layer(layer.source)

        if source.data.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
            LOGGER.warning(
                f"Cannot handle distance unit {source.data.crs().mapUnits()}, reprojecting to SWEREF99TM, this will only make sense in Sweden"
            )
            result = self._run_processing(
                "native:reprojectlayer",
                dict(
                    INPUT=source.data,
                    TARGET_CRS=QgsCoordinateReferenceSystem("EPSG:3006"),
                    OPERATION=None,
                    OUTPUT="memory:",
                ),
            )["OUTPUT"]
        else:
            result = source.data

        result = self._run_processing(
            "native:buffer",
            dict(
                INPUT=result,
                OUTPUT="memory:",
                DISTANCE=layer.distance,
                SEGMENTS=5,
                END_CAP_STYLE=0,
                JOIN_STYLE=0,
                MITER_LIMIT=2,
                DISSOLVE=False,
            ),
        )["OUTPUT"]

        if source.data.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
            result = self._run_processing(
                "native:reprojectlayer",
                dict(
                    INPUT=result,
                    TARGET_CRS=source.data.crs(),
                    OPERATION=None,
                    OUTPUT="memory:",
                ),
            )["OUTPUT"]

        return VectorData(original=source.original, data=result)

    def _execute_union_layer(self, layer: UnionLayer) -> VectorData:
        source_a = self._execute_layer(layer.source_a)
        source_b = self._execute_layer(layer.source_b)
        result = self._run_processing(
            "native:union",
            dict(
                INPUT=source_a.data,
                OVERLAY=source_b.data,
                OVERLAY_FIELDS_PREFIX="",
                OUTPUT="memory:",
            ),
        )
        LOGGER.warning(
            f"Union result contains {result['OUTPUT'].featureCount()} features"
        )
        return VectorData(original=source_a.original, data=result["OUTPUT"])

    def _execute_intersection_layer(self, layer: IntersectionLayer) -> VectorData:
        source_a = self._execute_layer(layer.source_a)
        source_b = self._execute_layer(layer.source_b)

        overlay = self._run_processing(
            "native:dissolve",
            dict(
                INPUT=source_b.data, FIELD=[], SEPARATE_DISJOINT=True, OUTPUT="memory:"
            ),
        )

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GeoJSON"
        QgsVectorFileWriter.writeAsVectorFormatV3(
            source_a.data,
            "/mnt/c/Users/jan/Downloads/a.geojson",
            self._project.transformContext(),
            opts,
        )
        QgsVectorFileWriter.writeAsVectorFormatV3(
            overlay["OUTPUT"],
            "/mnt/c/Users/jan/Downloads/b.geojson",
            self._project.transformContext(),
            opts,
        )

        if (
            source_a.data.geometryType() == QgsWkbTypes.PolygonGeometry
            and source_b.data.geometryType() == QgsWkbTypes.LineGeometry
        ):
            # special case as intersecting polygon and line never will give a result
            result = self._run_processing(
                "native:extractbylocation",
                dict(
                    INPUT=source_a.data,
                    INTERSECT=overlay["OUTPUT"],
                    PREDICATE=[0],
                    OUTPUT="memory:",
                ),
            )
        else:
            result = self._run_processing(
                "native:intersection",
                dict(
                    INPUT=source_a.data,
                    OVERLAY=overlay["OUTPUT"],
                    INPUT_FIELDS=[],
                    OVERLAY_FIELDS=[],
                    OVERLAY_FIELDS_PREFIX="",
                    OUTPUT="memory:",
                ),
            )
        LOGGER.warning(
            f"Intersection went from {source_a.data.featureCount()} to {result['OUTPUT'].featureCount()} features"
        )
        return VectorData(original=source_a.original, data=result["OUTPUT"])

    def _execute_difference_layer(self, layer: DifferenceLayer) -> VectorData:
        source_a = self._execute_layer(layer.source_a)
        source_b = self._execute_layer(layer.source_b)
        result = self._run_processing(
            "native:difference",
            dict(
                INPUT=source_a.data,
                OVERLAY=source_b.data,
                OUTPUT="memory:",
            ),
        )
        return VectorData(original=source_a.original, data=result["OUTPUT"])

    def _execute_action(self, action: Action) -> str:
        return getattr(self, f"_execute_{to_snake_case(action.__class__.__name__)}")(
            action
        )

    def _execute_select_action(self, action: SelectAction) -> str:
        layer = self._execute_layer(action.layer)
        layer.original.selectByIds(
            [
                f.attribute(layer.original.primaryKeyAttributes()[0])
                for f in layer.data.getFeatures()
            ],
            QgsVectorLayer.SetSelection,
        )

        if layer.original.selectedFeatureCount() == 1:
            return "Selected 1 item"
        return f"Selected {layer.original.selectedFeatureCount()} items"

    def _execute_add_to_map_action(self, action: AddToMapAction) -> str:
        layer = self._execute_layer(action.layer)
        self._project.addMapLayer(layer.data)

        return "Added data as a new layer to the map"

    def _execute_count_action(self, action: CountAction) -> str:
        layer = self._execute_layer(action.layer)

        if layer.data.featureCount() == 1:
            return "There is 1 matching feature"
        else:
            return f"There are {layer.data.featureCount()} matching features"

    def _run_processing(self, algorithm: str, parameters: dict) -> dict:
        LOGGER.warning(
            f"Running algorithm {algorithm} with parameters: {repr(parameters)}"
        )
        return processing.run(algorithm, parameters, feedback=self._feedback)
