# Converted by an OPENAI API call using model: gpt-3.5-turbo-1106
import logging
import io
import os
import subprocess

from openai.resources import Files

from gov_pnnl_goss.SpecialClasses import File
from gov_pnnl_goss.gridappsd.api.ConfigurationHandler import ConfigurationHandler
from gov_pnnl_goss.gridappsd.api.ConfigurationManager import ConfigurationManager
from gov_pnnl_goss.gridappsd.api.LogManager import LogManager
from gov_pnnl_goss.gridappsd.api.PowergridModelDataManager import PowergridModelDataManager
from gov_pnnl_goss.gridappsd.api.SimulationManager import SimulationManager
from gov_pnnl_goss.gridappsd.configuration.CIMDictionaryConfigurationHandler import PrintWriter
from gov_pnnl_goss.gridappsd.configuration.DSSAllConfigurationHandler import DSSAllConfigurationHandler
from gov_pnnl_goss.gridappsd.dto.LogMessage import ProcessStatus
from gov_pnnl_goss.gridappsd.dto.YBusExportResponse import YBusExportResponse
from gov_pnnl_goss.gridappsd.utils.GridAppsDConstants import GridAppsDConstants


class VnomExportConfigurationHandler(ConfigurationHandler):
    TYPENAME = "Vnom Export"
    SIMULATIONID = "simulation_id"
    DIRECTORY = "directory"
    MODELID = "model_id"
    ZFRACTION = "z_fraction"
    IFRACTION = "i_fraction"
    PFRACTION = "p_fraction"
    SCHEDULENAME = "schedule_name"
    LOADSCALINGFACTOR = "load_scaling_factor"

    def __init__(self, log_manager: LogManager = None):
        self.logger = log_manager
        self.log = LogManager(VnomExportConfigurationHandler.__class__.__name__)
        self.power_grid_model_data_manager = PowergridModelDataManager()
        self.config_manager = ConfigurationManager()  # config_manager
        self.simulation_manager = SimulationManager()


    def start(self):
        if self.config_manager is not None:
            self.config_manager.register_configuration_handler("Vnom Export", self)
        else:
            # TODO send log message and exception
            self.log.warn("No Config manager available for " + self.__class__.__name__)

    def generate_config(self, parameters: dict, out: io.FileIO, process_id, username):
        simulation_id = parameters.get("simulation_id")
        model_id = None
        simulation_dir = None

        if simulation_id:
            simulation_context = self.simulation_manager.get_simulation_context_for_id(simulation_id)
            parameters["i_fraction"] = str(simulation_context.get_request().get_simulation_config().get_model_creation_config().get_i_fraction())
            parameters["z_fraction"] = str(simulation_context.get_request().get_simulation_config().get_model_creation_config().get_z_fraction())
            parameters["p_fraction"] = str(simulation_context.get_request().get_simulation_config().get_model_creation_config().get_p_fraction())
            parameters["load_scaling_factor"] = str(simulation_context.get_request().get_simulation_config().get_model_creation_config().get_load_scaling_factor())
            parameters["schedule_name"] = simulation_context.get_request().get_simulation_config().get_model_creation_config().get_schedule_name()
            parameters["model_id"] = simulation_context.get_request().get_power_system_config().get_line_name()
            parameters["directory"] = simulation_context.get_simulation_dir()
            parameters["simulation_start_time"] = simulation_context.get_request().get_simulation_config().get_start_time()
            parameters["simulation_duration"] = simulation_context.get_request().get_simulation_config().get_duration()

            simulation_dir = File(simulation_context.get_simulation_dir())
        
        else:
            model_id = GridAppsDConstants.get_string_property(parameters, "model_id", None)
            simulation_id = process_id

            if model_id is None:
                raise Exception("Model Id or simulation Id not provided in request parameters.")

            simulation_dir = File(self.config_manager.get_configuration_property(GridAppsDConstants.GRIDAPPSD_TEMP_PATH), "models/" + model_id)
            parameters["i_fraction"] = GridAppsDConstants.get_double_property(parameters.get("i_fraction", 0))
            parameters["z_fraction"] = GridAppsDConstants.get_double_property(parameters.get("z_fraction", 0))
            parameters["p_fraction"] = GridAppsDConstants.get_double_property(parameters.get("p_fraction", 0))
            parameters["load_scaling_factor"] = GridAppsDConstants.get_double_property(parameters.get("load_scaling_factor", 1))
            parameters["schedule_name"] = GridAppsDConstants.get_string_property(parameters.get("schedule_name", ""))
            parameters["model_id"] = model_id
            parameters["directory"] = simulation_dir

        command_file = File(simulation_dir, "opendsscmdInput.txt")
        dss_base_file = File(simulation_dir, "model_base.dss")
        
        for key in parameters:
            self.log.debug(key + " = " + parameters.get(key))

        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Generating DSS base file")
        base_print_writer = PrintWriter(io.StringWriter())
        base_configuration_handler = DSSAllConfigurationHandler(self.logger, self.simulation_manager, self.config_manager)
        base_configuration_handler.generate_config(parameters, base_print_writer, simulation_id, username)

        if not dss_base_file.exists():
            raise Exception("Error: Could not create DSS base file to export Vnom matrix")

        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Finished generating DSS base file")
        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Generating commands file for opendsscmd")

        with open(command_file, "w") as file_writer:
            file_writer.write("redirect model_base.dss\n")
            file_writer.write("batchedit transformer..* wdg=2 tap=1\n")
            file_writer.write("batchedit regcontrol..* enabled=false\n")
            file_writer.write("batchedit isource..* enabled=false\n")
            file_writer.write("batchedit vsource..* pu=1.0\n")
            file_writer.write("batchedit load..* enabled=false\n")
            file_writer.write("batchedit generator..* enabled=false\n")
            file_writer.write("batchedit pvsystem..* enabled=false\n")
            file_writer.write("batchedit storage..* enabled=false\n")
            file_writer.write("batchedit capacitor..* enabled=false\n")
            file_writer.write("solve\n")
            file_writer.write("export voltages base_voltages.csv\n")

        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Finished generating commands file for opendsscmd")
        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Generating Y Bus matrix")

        process_service_builder = subprocess.Popen("opendsscmd " + command_file.getName(), cwd=simulation_dir, shell=True, stdout=subprocess.PIPE)
        process_service_builder.start()
        process_service_builder.wait(10)
        response = YBusExportResponse()
        vnom_path = File(os.path.abspath(simulation_dir), "base_voltages.csv")
        with open(vnom_path, 'r') as f:
            lines = f.readlines()
        response.set_vnom(lines)

        self.logger.debug(ProcessStatus.RUNNING, simulation_id, "Finished generating Vnom export")

        out.write(response)
