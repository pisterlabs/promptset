from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ..tools.plot import plot_trajectory, plot_scatter, plot_trajectory_and_scatter, plot_dynamic_trajectory, \
    plot_heatmap, plot_heatmap_density


class PlotTrajectorySchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")
    existing_map_file: str = Field(description="The existing map file path(a json file).Else is None.")


class PlotTrajectoryTool(BaseTool):
    name = "plot_trajectory"
    description = ('''
    This function plots the trajectories on a plotly map, with different trajectories
    for each unique 'uid' in the data.
    If existing_map_file is provided, it loads the existing map and adds the new trajectories.(else is None)
    If 'uid' column is not present, it plots all data
    as a single trajectory.
    ''')
    args_schema: Type[PlotTrajectorySchema] = PlotTrajectorySchema

    def _run(
            self,
            input_file: str,
            output_file: str,
            existing_map_file: str,
    ):
        """Use the tool."""
        return plot_trajectory.plot_trajectory(input_file,output_file,existing_map_file)


class PlotScatterSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")
    existing_map_file: str = Field(description="The path to an existing map file to update with new scatter points (default is None).")


class PlotScatterTool(BaseTool):
    name = "plot_scatter"
    description = ('''
    This function plots the scatter points on a plotly map, with different points
    for each unique 'uid' in the data. 
    If existing_map_file is provided, it loads the existing map and adds the new trajectories.(else is None)
    If 'uid' column is not present, it plots all data as a single set of points.''')
    args_schema: Type[PlotScatterSchema] = PlotScatterSchema

    def _run(
            self,
            input_file: str,
            output_file: str,
            existing_map_file: str,

    ):
        """Use the tool."""
        return plot_scatter.plot_scatter(input_file, output_file,existing_map_file)


class PlotTrajectoryAndScatterSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    trajectory_input_file: str = Field(
        description="The trajectory data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    scatter_input_file: str = Field(
        description="The scatter point data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")
    existing_map_file: str = Field(description="The path to an existing map file to update with new scatter points ("
                                               "default is None).")


class PlotTrajectoryAndScatterTool(BaseTool):
    name = "plot_trajectory_and_scatter"
    description = ('''
    This function plots the trajectories and scatter points on a plotly map, with different colors
    for each unique 'uid' in the data. If 'uid' column is not present, it plots all data
    as a single trajectory and a single set of scatter points.
    If existing_map_file is provided, it loads the existing map and adds the new trajectories.(else is None)''')
    args_schema: Type[PlotTrajectoryAndScatterSchema] = PlotTrajectoryAndScatterSchema

    def _run(
            self,
            trajectory_input_file: str,
            scatter_input_file: str,
            output_file: str,
            existing_map_file: str,
    ):
        """Use the tool."""
        return plot_trajectory_and_scatter.plot_trajectory_and_scatter(
            trajectory_input_file, scatter_input_file, output_file,existing_map_file)


class PlotDynamicTrajectorySchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")


class PlotDynamicTrajectoryTool(BaseTool):
    name = "plot_dynamic_trajectory"
    description = "Dynamically plot the trajectories on an interactively plotly map."
    args_schema: Type[PlotDynamicTrajectorySchema] = PlotDynamicTrajectorySchema

    def _run(
            self,
            input_file: str,
            output_file: str,
    ):
        """Use the tool."""
        return plot_dynamic_trajectory.plot_dynamic_trajectory(input_file, output_file)


class PlotHeatmapSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(description="The data file path to be processed.(must contain \"lat\", \"lng\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")


class PlotHeatmapTool(BaseTool):
    name = "plot_heatmap"
    description = """This function plots the heatmap from a series of trajectories on a plotly map.
    This function calculates the density values from the intensity of points."""
    args_schema: Type[PlotHeatmapSchema] = PlotHeatmapSchema

    def _run(
            self,
            input_file: str,
            output_file: str,
    ):
        """Use the tool."""
        return plot_heatmap.plot_heatmap(input_file, output_file)


class PlotHeatmapDensitySchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_file: str = Field(
        description="The data file path to be processed.(must contain \"lat\", \"lng\", \"density\" rows)")
    output_file: str = Field(description="The output graph file path(a json file).")


class PlotHeatmapDensityTool(BaseTool):
    name = "plot_heatmap_density"
    description = """This function plots the heatmap from a series of trajectories on a plotly map.
    This function requires the density values to be stored in the input file."""
    args_schema: Type[PlotHeatmapDensitySchema] = PlotHeatmapDensitySchema

    def _run(
            self,
            input_file: str,
            output_file: str,
    ):
        """Use the tool."""
        return plot_heatmap_density.plot_heatmap_density(input_file, output_file)
