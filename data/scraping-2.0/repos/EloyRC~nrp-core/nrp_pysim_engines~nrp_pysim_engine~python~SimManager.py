import time


class SimulatorManager(object):
    """
    This class receives the information that a simulator needs from an EngineScript,
    and it then starts and manages a simulator via its Python API.
    """

    def __init__(self, configuration):
        super(SimulatorManager, self).__init__()

        world_file = configuration["WorldFileName"]
        start_visualizer = bool(configuration["Visualizer"])
        simulator_type = configuration["Simulator"]

        self.time_step = configuration["EngineTimestep"]

        extra_config = configuration["EngineExtraConfigs"]

        self.sim_interface = None
        if simulator_type == "Opensim":
            from .OpensimLib import OpensimInterface
            self.sim_interface = OpensimInterface(world_file, start_visualizer, self.time_step, extra_config)
        elif simulator_type == "OpenAI":
            from .OpenAIGymLib import OpenAIInterface
            self.sim_interface = OpenAIInterface(world_file, start_visualizer, self.time_step, extra_config)
        elif simulator_type == "Mujoco":
            from .MujocoLib import MujocoInterface
            self.sim_interface = MujocoInterface(world_file, start_visualizer, self.time_step, extra_config)
        elif simulator_type == "Bullet":
            from .BulletLib import BulletInterface
            self.sim_interface = BulletInterface(world_file, start_visualizer, self.time_step, extra_config)
        else:
            raise Exception(f'Simulator {simulator_type} is not installed')
            
    def reset(self):
        """
        Reset the simulation, it is connected by the "server_callbacks.py"
        """
        return self.sim_interface.reset()

    def shutdown(self):
        """
        Shutdown the simulation, it is connected by the "server_callbacks.py"
        """
        self.sim_interface.shutdown()

    def run_step(self, action, timestep_ns):
        """
        Obtain parameters from the engine script and run the simulation step by step

        :param action: the control parameters for the simulation
        :type action: list
        :param timestep_ns: time step length of the simulation (nanosecs)
        :type timestep_ns: int
        """
        self.sim_interface.run_one_step(action, timestep_ns)

    def get_model_properties(self, datapack_type):
        """
        Obtain devices list

        :param datapack_type: data type of the required device
        :type datapack_type: str
        """
        return self.sim_interface.get_model_properties(datapack_type)

    def get_model_all_properties(self, datapack_type):
        """
        Obtain all devices data of a special type

        :param datapack_type:  (string): data type of required devices
        :type datapack_type: str
        """
        return self.sim_interface.get_model_all_properties(datapack_type)

    def get_model_property(self, datapack_name, datapack_type):
        """
        Obtain data of a device based on its name

        :param datapack_name: name of the required device
        :type datapack_type: str
        :param datapack_type: data type of the required device
        :type datapack_type: str
        """
        return self.sim_interface.get_model_property(datapack_name, datapack_type)

    def get_sim_time(self):
        return self.sim_interface.get_sim_time()
