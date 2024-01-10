import os

from .compression_tool import CompressionTool
from .filtering_tool import FilteringTool
from .plot_trajectory import PlotTrajectoryTool, PlotScatterTool, PlotTrajectoryAndScatterTool, PlotHeatmapDensityTool, \
    PlotHeatmapTool, PlotDynamicTrajectoryTool
from .stop_detection_tool import StopDetectionTool
from .json_tool import JsonTool
from .home_location_tool import HomeLocationTool
from .max_distance_tool import MaxDistanceTool
from .gyration_tool import RadiusGyrationTool, KRadiusGyrationTool
from .jump_lengths_tool import JumpLengthsTool
from .recency_rank_tool import RecencyRankTool
from .table_reader_tool import TableReaderTool
from .geo_decode_tool import GeoDecodeTool
from .python_repl_tool import CustomPythonREPLTool
from .rag_tool import RAGTool
from typing import List
from langchain.tools import BaseTool

from .weather_tool import WeatherTool

from ..tools.rag import retriever


def collect_tools():
    tools: List[BaseTool] = [CompressionTool(),
                             FilteringTool(),
                             StopDetectionTool(),
                             HomeLocationTool(),
                             JsonTool(),
                             MaxDistanceTool(),
                             RadiusGyrationTool(),
                             KRadiusGyrationTool(),
                             JumpLengthsTool(),
                             RecencyRankTool(),
                             TableReaderTool(),
                             CustomPythonREPLTool(),
                             PlotTrajectoryTool(),
                             PlotScatterTool(),
                             PlotHeatmapDensityTool(),
                             PlotHeatmapTool(),
                             PlotDynamicTrajectoryTool(),
                             PlotTrajectoryAndScatterTool(),
                             GeoDecodeTool(),
                             WeatherTool(),
                             ]
    SERPAPI_API_KEY = "SERPAPI_API_KEY"
    if SERPAPI_API_KEY in os.environ and os.environ[SERPAPI_API_KEY]:
        from .search_tool import SearchTool
        tools.append(SearchTool)
    GAO_DE_API_KEY = "GAO_DE_API_KEY"
    if GAO_DE_API_KEY in os.environ and os.environ[GAO_DE_API_KEY]:
        from .reverse_geodecode_tool import ReverseGeodecodeTool
        tools.append(ReverseGeodecodeTool())
        from .POI_search_tool import POISearchTool
        tools.append(POISearchTool())

    db = retriever.load_db("./faiss", "db")
    retriever.set_db(db)
    tools.append(RAGTool())

    for i in range(len(tools)):
        tools[i].handle_tool_error = lambda \
                e: "Error occurs, you may check the existance of file and use table reader to check the data: " + str(e)
    return tools
