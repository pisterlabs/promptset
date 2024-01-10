from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
from collections import deque

import pdb

import airsim

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class Env(object):
    """
    Stripped down version from OpenaiGym
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None
    ser_msg_length = 0

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def serialize_obs(self):
        """Returns a function that allows you to serialize each observation as a multipart"""
        raise NotImplementedError
    
class SIM_f110Env(Env):
    """
    Implements a Gym Environment & neccessary funcs for the F110 Autonomous Car on Microsfot Airsim
    """
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True) 
        self.history = deque(maxlen=500) #for reversing during reset

        #GYM Properties (set in subclasses)
        self.observation_space = ['lidar', 'steer', 'img']
        self.action_space = ['angle', 'speed']
    
    ###########GYM METHODS##################################################

    def transformLidar(self, lidarData):
        """ Transform Lidar pointcloud into ranges array
        """
        pc = lidarData.point_cloud
        pcnp = np.array(pc)
        pcnp = np.reshape(pcnp, (-1, 3))
        pcnp = pcnp[..., 0:2]
        return pcnp

    def _get_obs(self):
        #Get Camera imgs
        imgs = self._get_imgs()

        #Get LiDAR reading & sort returned pointclouds according to their angle (from left)
        while True:
            lidarData = self.client.getLidarData()
            if len(lidarData.point_cloud) >= 3:
                break
        lidarData = self.process_lidar(lidarData)

        #Get steer data
        steer = {"angle": 0.0, "steering_angle_velocity": 0.0, "speed": 0.0}
        latest_dict = {'lidar_pc': lidarData, 'steer': steer, 'img':imgs}
        self.add_to_history(steer)
        return latest_dict

    def reset(self, **kwargs):
        """
        Reset to initial position
        """
        self.client.reset()
        time.sleep(1)
        return self._get_obs()

    def get_reward(self):
        """
        TODO:Implement reward functionality
        """
        return 0

    def step(self, action):
        """
        Action should be a steer_dict = {"angle":float, "speed":float}
        """
        car_controls = airsim.CarControls()
        car_controls.throttle = action.get("speed")
        car_controls.steering = action.get("angle")

        #execute action
        self.client.setCarControls(car_controls)
        time.sleep(0.01)

        #get reward & check if done & return
        obs = self._get_obs()
        reward = self.get_reward()
        done = self.tooclose()
        info = {}
        return obs, reward, done,info
    
    def tooclose(self):
        """
        Uses latest_obs to determine if we are too_close (currently uses LIDAR)
        """
        return False

    ###########EXTRA METHODS##################################################

    def _get_imgs(self, labels=["front_center"]):
        label_to_func = lambda lbl: airsim.ImageRequest(lbl, airsim.ImageType.Scene, False, False)
        bytestr_to_np = lambda rep: np.fromstring(rep.image_data_uint8, dtype=np.uint8).reshape(rep.height, rep.width, -1)
        responses = self.client.simGetImages(list(map(label_to_func, labels)))
        images = list(map(bytestr_to_np, responses))
        return images
    
    def add_to_history(self, steer):
        if abs(steer["angle"]) > 0.05 and steer["angle"] < -0.05:
            for i in range(40):
                self.history.append(steer)

    def vis_lidarpc(self, lidar):
        """ Visualize a lidarPointcloud
        Expects lidar data in 2d array [x, y]
        """
        # lidar_frame = np.ones((500, 500, 3)) * 75
        lidar_frame = np.zeros((500, 500, 3))
        cx = 250
        cy = 250
        rangecheck = lambda x, y: abs(x) < 1000. and abs(y) < 1000.
        for i in range(lidar.shape[0]):
            x = lidar[i, 0]
            y = lidar[i, 1]
            if (rangecheck(x, y)):
                scaled_x = int(cx + x*50) 
                scaled_y = int(cy - y*50)
                cv2.circle(lidar_frame, (scaled_x, scaled_y), 1, (255, 255, 255), -1)
        cv2.imshow("lidarframe", lidar_frame)
        cv2.waitKey(1)

    def pc_to_np(self, lidarData):
        #Get out the x,y
        pc = lidarData.point_cloud
        pcnp = np.array(pc)
        pcnp = np.reshape(pc, (-1, 3))
        pcnp = pcnp[..., 0:2]
        out = np.empty_like(pcnp)
        out[:, 0] = pcnp[:, 1]
        out[:, 1] = pcnp[:, 0]
        return out

    def process_lidar(self, lidarData):
        """ Process pointcloud lidar data into a ranges array 
        """
        lidarData = self.pc_to_np(lidarData)

        #sort lidarData by radians measurmeent
        pt_to_rad = lambda pt : math.atan2(pt[1], pt[0])
        lidar_out = np.empty_like(lidarData)
        rads = []
        
        # for i in range(pcnp.shape[0]):
        #     rad, rang = self.xy_to_radrange(pcnp[i, 0], pcnp[i, 1])
        #     rads.append((rad, rang))
        # rads.sort(key=lambda x:x[1])

        # #LIDAR info - angle_min: -1.5707950592 angle_incr: 0.027 
        # lidarData = self.rads_to_ranges(rads)
        # return pcnp