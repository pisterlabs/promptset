import socket
import random
import os
import pathlib

from datetime import datetime
from multiprocessing import Process
from openai_ros2.utils import ut_generic

from ament_index_python.packages import get_package_prefix
from launch import LaunchService, LaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def startLaunchServiceProcess(launchDesc):
    """Starts a Launch Service process. To be called from subclasses.
    Args:
         launchDesc : LaunchDescription obj.
    """
    # Create the LauchService and feed the LaunchDescription obj. to it.
    launchService = LaunchService()
    launchService.include_launch_description(launchDesc)
    process = Process(target=launchService.run)
    # The daemon process is terminated automatically before the main program exits,
    # to avoid leaving orphaned processes running
    process.daemon = True
    process.start()

    return process


def isRosDomainInUse(domain_id: int):
    # For port calculation, refer to https://fast-rtps.docs.eprosima.com/en/latest/advanced.html#listening-locators
    # Only works for fast-rtps dds for now, also maybe opensplice
    port = 7400 + 250 * domain_id
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # period to see if any data comes into the udp port (1.0 means wait for 1s for data to come)
        sock.settimeout(1.0)
        try:
            sock.bind(('', port))
            data, addr = sock.recvfrom(64)
            print(f"[{domain_id}]data: {data}, addr: {addr}")
            return True
        except socket.timeout:
            return False
        except OSError as ex:
            if ex.errno == 98:  # "port in use" error code
                return True
            return False


def isPortInUse(port):
    """Checks if the given port is being used.
    Args:
        port(int): Port number.
    Returns:
        bool: True if the port is being used, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket1:
        return socket1.connect_ex(('localhost', port)) == 0


def getExclusiveNetworkParameters():
    """Creates appropriate values for ROS_DOMAIN_ID and GAZEBO_MASTER_URI.
    Returns:
        Dictionary {ros_domain_id (string), ros_domain_id (string)}
    """

    randomPortROS = random.randint(0, 200)
    randomPortGazebo = random.randint(10000, 15000)
    while isRosDomainInUse(randomPortROS):
        print("Randomly selected ROS Domain is already in use, retrying.")
        randomPortROS = random.randint(0, 200)

    while isPortInUse(randomPortGazebo):
        print("Randomly selected gazebo port is already in use, retrying.")
        randomPortGazebo = random.randint(10000, 15000)

    # Save network segmentation related information in a temporary folder.
    cwd: str = os.getcwd()
    tempPath = os.path.join(cwd, 'tmp/openai-ros-2/running/')
    pathlib.Path(tempPath).mkdir(parents=True, exist_ok=True)

    # Remove old tmp files.
    ut_generic.cleanOldFiles(tempPath, ".log", days=2)

    filename = datetime.now().strftime('running_since_%H_%M__%d_%m_%Y.log')

    file = open(tempPath + '/' + filename, 'w+')
    file.write(filename + '\nROS_DOMAIN_ID=' + str(randomPortROS) \
               + '\nGAZEBO_MASTER_URI=http://localhost:' + str(randomPortGazebo))
    file.close()

    return {'ros_domain_id': str(randomPortROS),
            'gazebo_master_uri': "http://localhost:" + str(randomPortGazebo)}


def generate_launch_description_lobot_arm(use_gui: bool = False):
    # Get simulation package path
    sim_share_path = get_package_share_directory('arm_simulation')

    # Launch param server
    params_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sim_share_path, 'launch',
                                                   'params_server.launch.py')),
    )
    # Launch arm spawner
    arm_spawner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sim_share_path, 'launch',
                                                   'gazebo_spawn_arm.launch.py')),
        launch_arguments={'gym': 'True', 'gui': f'{use_gui}'}.items()
    )
    return LaunchDescription([
        params_server,
        arm_spawner
    ])


def set_network_env_vars():
    network_params = getExclusiveNetworkParameters()
    os.environ["ROS_DOMAIN_ID"] = network_params.get('ros_domain_id')
    os.environ["GAZEBO_MASTER_URI"] = network_params.get('gazebo_master_uri')
    os.environ["IGN_PARTITION"] = network_params.get('ros_domain_id')
    print("******* Exclusive network segmentation *******")
    print("ROS_DOMAIN_ID=" + network_params.get('ros_domain_id'))
    print("GAZEBO_MASTER_URI=" + network_params.get('gazebo_master_uri'))
    print("IGN_PARTITION=" + network_params.get('ros_domain_id'))
    print("")
