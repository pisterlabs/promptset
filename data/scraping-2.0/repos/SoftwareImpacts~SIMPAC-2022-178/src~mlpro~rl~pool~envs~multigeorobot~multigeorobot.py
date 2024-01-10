## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : multigeorobot.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-19  0.0.0     MRD      Creation
## -- 2021-12-19  1.0.0     MRD      Released first version
## -- 2022-02-25  1.0.1     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-04-29  1.0.2     MRD      Wrap the environment with WrEnvGYM2MLPro
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-04-29)

This module provides an environment for multi geometry robot.
"""

from mlpro.rl.models import *
import mlpro
import rospy
import subprocess
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

class MultiGeo(WrEnvGYM2MLPro):
    """
    This module provides an environment for multi geometry robot.
    """

    C_NAME      = 'MultiGeo'
    C_LATENCY   = timedelta(0,5,0)
    C_INFINITY  = np.finfo(np.float32).max  

    def __init__(self, p_seed=0, p_logging=True):
        roscore = subprocess.Popen('roscore')
        rospy.init_node('multi_geo_robot_training', anonymous=True, log_level=rospy.WARN)

        LoadYamlFileParamsTest(rospackage_name="multi_geo_robot_training",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="multi_geo_robot.yaml")  

        ros_ws_path = mlpro.rl.pool.envs.multigeorobot.__file__.replace("/__init__.py", "")
        rospy.set_param('ros_ws_path', ros_ws_path)
        # Init OpenAI_ROS ENV
        task_and_robot_environment_name = rospy.get_param(
        '/multi_geo_robot/task_and_robot_environment_name')

        max_step_episode = rospy.get_param(
        '/multi_geo_robot/max_iterations')  

        env = StartOpenAI_ROS_Environment(task_and_robot_environment_name, max_step_episode)
        env.seed(p_seed)       
        
        super().__init__(p_gym_env=env)
    
## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        obs = p_state.get_values()
        close = np.allclose(a=obs[:3],
                            b=obs[3:],
                            atol=0.1)

        if close:
            self._state.set_terminal(True)

        return close