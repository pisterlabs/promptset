#!/usr/bin/env python
from gym import spaces
from openai_ros.robot_envs import double_bebop2_env
from openai_ros.openai_ros_common import ROSLauncher
from gym.envs.registration import register
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
from geometry_msgs.msg import Vector3, Pose
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import time
import numpy as np
from pathlib import Path
import math

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
MAX_STEP = 1000 # Can be any Value

register(
        id='DoubleBebop2Env-v0',
        entry_point='openai_ros.task_envs.bebop2.double_bebop2_task:DoubleBebop2TaskEnv',
        max_episode_steps=MAX_STEP,
    )

class DoubleBebop2TaskEnv(double_bebop2_env.DoubleBebop2Env):
    def __init__(self):


        # Lancement de la simulation
        ROSLauncher(rospackage_name="rotors_gazebo", launch_file_name="mav_train.launch", 
                    ros_ws_abspath=str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))
        
        # On charge methodes et atributs de la classe mere
        super(DoubleBebop2TaskEnv, self).__init__()

        self.observation_space = spaces.Box(low = np.array([-30,-30,-30]), high = np.array([30,30,30]), dtype = np.float32)
        self.action_space = spaces.Box(low = np.array([-1,-1,-1]), high = np.array([1,1,1]), dtype = np.float32)



    def _set_init_pose(self):
        """Sets the Robot in its init pose
        Appelée lorsqu'on reset la simulation
        """
        # On reset cmd_vel
        self.publish_cmd("both", 0,0,0,0)
        # self.gazebo.pauseSim()
        # self.gazebo.resetSim()
        # self.gazebo.unpauseSim()
        # Il est important dans notre cas de reset_pub juste apres le resest, c'est pour ca on reset (c'est pas grv si on resset 2 fois)
        # il est necessaire de reset deux fois pour que cela soit pris en compte 


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """


        self.reset_pub()
        rospy.sleep(0.1)
        self.gazebo.unpauseSim()

        # Boucle de sécurité pour éviter les cas foireux
        while True:
            rospy.sleep(0.1)
            self.takeoff()
            L_alt = self.L_odom.pose.pose.position.z
            R_alt = self.R_odom.pose.pose.position.z
            dist_x = abs(self.L_odom.pose.pose.position.x - self.R_odom.pose.pose.position.x)
            dist_y = abs(self.L_odom.pose.pose.position.y - self.R_odom.pose.pose.position.y)
            dist_z = abs(L_alt - R_alt) 

            distance =  self.compute_dist(dist_x, dist_y, dist_z) 
            if distance >1.02 or L_alt < 0.25 or R_alt < 0.25:

                rospy.logerr(f"Problème detecté, reset : dist : {distance:.3f}, L_alt : {L_alt:.2f}, R_alt : {R_alt:.2f}")

                self.gazebo.pauseSim()
                self._reset_sim()
                self.reset_pub()
                self.gazebo.unpauseSim()
            else:
                break

        self.gazebo.pauseSim()
        self.number_step = 0


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        On utilise PPO continue, les actions seront un vecteur de taille 4 continue
        action = [linear.x, linear.y, linear.z, angular.z]
        On fait bouger le R_bebop qui suit le L_bebop
        """
        lin_x, lin_y, lin_z = action
        self.publish_cmd("R_bebop2",lin_x,lin_y,lin_z)
        self.number_step += 1
        

    def _get_obs(self):
        """
        Obsevations : 
        -   distance between drones obs[0...2]
        -   L drone speed, R drone speed : obs[3...6], obs[6...9] 
        -   Euler orientaiton of both drones

        """
        # Distance
        dist_x = abs(self.L_odom.pose.pose.position.x - self.R_odom.pose.pose.position.x)
        dist_y = abs(self.L_odom.pose.pose.position.y - self.R_odom.pose.pose.position.y)
        dist_z = abs(self.L_odom.pose.pose.position.z - self.R_odom.pose.pose.position.z)



        observation = np.array([dist_x, dist_y, dist_z])
        return  observation
    



    def wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))



    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        
        L'episode se finit si: 
        -   l'un des drones se retourne.
        -   les drones sont trop éloignés ou trop proche
        -   le drone a fait un bon nombre de step sans perdre
        """
        done = False

        #Check if one of the two UAV is upside down
        # L_roll = observations[11]
        # L_pitch = observations[12]

        # R_roll = observations[14]
        # R_pitch = observations[15]
        # L_yaw, R_yaw = observations[13], observations[16]
        
        # yaw_error = abs(L_yaw-R_yaw)

        # if abs(L_pitch) >= 1.57 or abs(L_roll) >= 1.57:
        #     rospy.logdebug("Le L_bebop s'est retourné")
        #     done = True

        # if abs(R_pitch) >= 1.57 or abs(R_roll) >= 1.57:
        #     rospy.logdebug("Le R_bebop s'est retourné")
        #     done = True
        

        # Check the distance between the drone
        dist_x, dist_y, dist_z = observations[0:3]
        # distance = self.compute_dist(dist_x, dist_y)

        # if distance > 1.5 or distance < 0.5: done = True
        if dist_x > 0.2: done = True
        if dist_y > 1.5: done = True
        if dist_y < 0.5: done = True
        if dist_z > 0.2: done = True
        # if yaw_error > 0.53 : done = True # ~ 30 degrees 
        if self.L_odom.pose.pose.position.z < 0.2 or self.R_odom.pose.pose.position.z < 0.2 : done = True

        return done
    

    def _compute_reward(self, observations, done):
        """ On veut que les drones soient synchronisé avec un espace de 1 metre entre eux
            Par la suite on voudra aussi que le drones esquives les obstacles (pas pour l'instant)
        On utilisera une reward linéiar en fonction de la distance entre les drones
        """
        # System rewards 1 2 ou 3 
        reward = self.reward_system0bis(observations, done)
        return reward
        
    # Internal TaskEnv Methods


    def compute_dist(self,*args):
        d = np.array(args)
        dist = np.sum(d**2)**0.5
        return dist


    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw


##### REWARD SYSTEM : 
    def reward_system0(self, observations, done): 

        dist_x, dist_y, dist_z = observations[0:3]
        distance = self.compute_dist(dist_x, dist_y)
        reward = 0

        if done : 
            if distance > 2: 
                reward = rospy.get_param("overdist_reward",-350)
            if distance < 0.6:
                reward = rospy.get_param("near_distance_end", -350)
            elif dist_z > 0.2:
                reward = rospy.get_param("overalt_reward",-100)
            
        if not done: 
            if (1 - distance)  > 0.15:
                reward  = rospy.get_param("good_reward", 15)
            elif distance < 0.65: 
                reward =  rospy.get_param("near_reward",-30)
            
            # Encouraging number of step
            reward +=2
        
        return reward
        
    def reward_system0bis(self, observations, done): 

        dist_x, dist_y, dist_z = observations[0:3]
        if done : 
            if dist_x >=0.2:
                reward = -200
            
            if dist_y >= 1.5:
                reward = -300
            elif dist_y <= 0.5:
                reward = -300
            
            if dist_z >= 0.2:
                reward = -300
        
        else:
            reward = 0
            if dist_x < 0.1:
                reward  += 2

            if abs(dist_y - 1) < 0.1:
                reward += 4

            if dist_z < 0.1:
                reward += 2
        
        return reward




    def reward_system1(self, observations, done):
        L_roll = observations[11]
        L_pitch = observations[12]

        R_roll = observations[14]
        R_pitch = observations[15]

        dist_x, dist_y, dist_z = observations[0:3]
        distance = self.compute_dist(dist_x, dist_y)
        reward = 0

        if not done: 
            if abs(1 - distance) < 0.1: 
                reward += 10
            elif distance > 1:
                #Pourcentage
                reward += -100*(distance - 1)/0.5
                # assert reward <= 0
                
            elif distance < 1:
                # Pourcentage 
                reward += -300*(distance - 0.5)/0.5
                # assert reward <= 0

            if dist_z > 0.1:
                reward -= 300 
            else: reward +=30

            # if self.L_pose.position.z < 0.2 or self.R_pose.position.z < 0.2 :
            #     reward -= 30
            # else:
            #     reward += 3

        
        else:
            if distance < 0.5:
                # Si l'episode se termine acvec les drones trop proche (pret a se cogner)
                reward += -500
            elif distance > 1.5:
                # Si les drones se sont trop éloigné
                reward += -200
            else:
                reward += 10
            
            if dist_z > 0.1:
                reward -= 300 
            else: reward +=30
            

            if abs(L_pitch) >= 1.57 or abs(L_roll) >= 1.57 or abs(R_pitch) >= 1.57 or abs(R_roll) >= 1.57:
                reward += -500
            else: 
                reward += 10

            
            if self.number_step >= MAX_STEP:
                #Dans ce cas on aura fini l'épiosde sans acro
                reward += 1000
        return reward
    
    def reward_system2(self, observations, done):

        #end episode
        out_reward = -150
        too_near_reward = -200
        total_bad_altitude_reward = -150
        good_yaw = 15
        bad_yaw = -20
        good_distance_reward = 20
        good_altitude_reward = 20
        bad_distance_reward = -40
        bad_altitude_reward = -40
        step_reward = 5


        dist_x, dist_y, dist_z = observations[0:3]
        L_yaw, R_yaw = observations[13], observations[16]
        
        yaw_error = abs(L_yaw-R_yaw)

        distance = self.compute_dist(dist_x, dist_y)
        if done :
            reward = 0
            if distance > 1.5:
                reward += out_reward
            if distance < 0.5:
                reward += too_near_reward
            if dist_z > 0.2:
                reward += total_bad_altitude_reward
            return reward

        else:
            reward = 0
            if abs(distance -1 ) < 0.1:
                reward += good_distance_reward
            else : 
                reward += bad_distance_reward
            if dist_z < 0.1: 
                reward += good_altitude_reward
            else:
                reward += bad_altitude_reward

            if yaw_error < 0.2 : # ~11 degrees
                reward += good_yaw
            else:
                reward += bad_yaw

            reward += step_reward
            return reward

    def reward_system3(self, observations, done):
        # Ah ouai ?
        out_reward = -400
        too_near_reward = -500
        total_bad_altitude_reward = -350
        good_yaw = 5
        bad_yaw = -10
        good_distance_reward = 10
        good_altitude_reward = 10
        step_reward = 3

        speed_penalty = -50


        dist_x, dist_y, dist_z = observations[0:3]
        Lvx, Lvy, Lvz, Laz, Rvx, Rvy, Rvz, Raz = observations[3:11]
        L_yaw, R_yaw = observations[13], observations[16]
        
        yaw_error = abs(L_yaw-R_yaw)

        distance = self.compute_dist(dist_x, dist_y)
        if done :
            reward = 0
            if distance > 1.5:
                reward += out_reward
            if distance < 0.5:
                reward += too_near_reward
            if dist_z >= 0.2:
                reward += total_bad_altitude_reward
            return reward

        else:
            reward = 0
            if abs(distance -1 ) < 0.15:
                reward += good_distance_reward

            if dist_z < 0.1: 
                reward += good_altitude_reward

            if yaw_error < 0.2 : # ~11 degrees
                reward += good_yaw
            else:
                reward += bad_yaw
            
            # Il faut que les vitesses soient les memes (les drones vont au meme endroit) (sauf si ils sont trop proche)
            if distance > 0.8: 
                if Lvx*Rvx < 0  and abs(Lvx - Rvx) > 0.05:
                    reward += speed_penalty

                if Lvy*Rvy < 0  and abs(Lvy - Rvy) > 0.05:
                    reward += speed_penalty

                if Lvz*Rvz < 0  and abs(Lvz - Rvz) > 0.05:
                    reward += speed_penalty

                if Laz*Raz < 0  and abs(Laz - Raz) > 0.05:
                    reward += speed_penalty

            reward += step_reward
            return reward


            
            
            




