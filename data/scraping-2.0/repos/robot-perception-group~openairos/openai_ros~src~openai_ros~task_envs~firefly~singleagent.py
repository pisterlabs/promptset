#!/usr/bin/env python


# -*- coding: utf-8 -*-

import numpy as np

import rospy
import tf
from geometry_msgs.msg import Point, PointStamped,Pose, PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import Float64, Int16


from gym import spaces
from openai_ros.robot_envs import singleagentSim_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

gt_joints={  'actor::actor_pose':0,
             'actor::Head':7,
             'actor::LeftArm':8,
             'actor::RightArm':11,
             'actor::LeftForeArm':9,
             'actor::RightForeArm':12,
             'actor::LeftFingerBase':10,
             'actor::RightFingerBase':13,
             'actor::LeftUpLeg':1,
             'actor::RightUpLeg':4,
             'actor::LeftLeg':2,
             'actor::RightLeg':5,
             'actor::LeftFoot':3,
             'actor::RightFoot':6,
             }


weight = {0:0.025,    1:0.075,  2:0.025, 3:0.025, 4:0.05,  5:0.05,  6:0.2,  7:0.2,  8:0.025,  9:0.025,  10:0.05,     11:0.05,     12:0.15,    13:0.15}

gt_joints={'actor::RightFoot':6,'actor::RightLeg':5,'actor::RightUpLeg':4,'actor::LeftUpLeg':1,'actor::LeftLeg':2,'actor::LeftFoot':3,\
'actor::RightFingerBase':13,'actor::RightForeArm':12,'actor::RightArm':11,'actor::LeftArm':9,'actor::LeftForeArm':10,\
'actor::LeftFingerBase':8,'actor::actor_pose':0,'actor::Head':7}


gt_joints_names = list(gt_joints.keys())
num_joints = len(gt_joints_names)

class DeepFollowEnv(singleagentSim_env.FireflyMultiAgentGTEnv):
    def __init__(self,**kwargs):
        """
        This Task Env is designed for MAVs to track a human target
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        rospy.logwarn('Entered Single Agent Testing ENV')
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/firefly/config",
                               yaml_file_name="formation.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeepFollowEnv, self).__init__(**kwargs)

        # Get WorkSpace Cube Dimensions
        self.work_space_max = rospy.get_param("/work_space/max")
        self.work_space_min = rospy.get_param("/work_space/min")

        # Get Velocity Limits
        self.max_v =  rospy.get_param("/work_space/max_v")
        self.min_v =  rospy.get_param("/work_space/min_v")

        self.image_width = rospy.get_param("/image_width")
        self.image_height = rospy.get_param("/image_height")

        low = []
        
        '''
        Policy Network Inputs: Lower Bounds
        '''
        #dx,dy,dz,
        #vx,vy,vz,dbearing(2D),dperson_orientation((2D)
        low = np.concatenate((low,np.array([self.work_space_min,self.work_space_min,self.work_space_min,\
                                            self.min_v,self.min_v,self.min_v,-1,-1,-1,-1]))) 
        '''
        Value Network Inputs: Lower Bounds
        '''
        #14 joints (x,y,z)
        #x,y,z MAV
        #vx,vy,vz MAV
        #pvx,pvy,pvz, Person
        #yaw,person_orientation        
        low = np.concatenate((low,np.tile(np.array([self.work_space_min,self.work_space_min,self.work_space_min]),14))) #JOINTS
        low = np.concatenate((low,np.array([self.work_space_min,self.work_space_min,self.work_space_min,
                                            self.min_v,self.min_v,self.min_v,
                                            self.min_v,self.min_v,self.min_v,\
                                            -1,-1,-1,-1])))         

        '''
        Policy Network Inputs: Upper Bounds
        '''        
        #dx,dy,dz,
        #vx,vy,vz,dbearing(2D),dperson_orientation((2D)     
        high=[]
        high = np.concatenate((high,np.array([self.work_space_max,self.work_space_max,self.work_space_max,\
                                              self.max_v,self.max_v,self.max_v,1,1,1,1]))) 
        '''
        Value Network Inputs: Upper Bounds
        '''
        #14 joints (x,y,z)
        #x,y,z MAV
        #vx,vy,vz MAV
        #pvx,pvy,pvz, Person
        #yaw,person_orientation                 
        high = np.concatenate((high,np.tile(np.array([self.work_space_max,self.work_space_max,self.work_space_max]),14))) #JOINTS
        high = np.concatenate((high,np.array([self.work_space_max,self.work_space_max,self.work_space_max,\
                                              self.max_v,self.max_v,self.max_v,
                                              self.max_v,self.max_v,self.max_v,\
                                              1,1,1,1]))) 

        # Defining Observation Space
        self.observation_space = spaces.Box(low,high)

        # Defining Action Space
        self.control_x_max = rospy.get_param("/work_space/vx_max")
        self.control_x_min = rospy.get_param("/work_space/vx_min")
        self.control_y_max = rospy.get_param("/work_space/vy_max")
        self.control_y_min = rospy.get_param("/work_space/vy_min")
        self.control_z_max = rospy.get_param("/work_space/vz_max")
        self.control_z_min = rospy.get_param("/work_space/vz_min")

        self.ACTION_TYPE = rospy.get_param("/actions")
        if self.ACTION_TYPE=="discrete":
            self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(np.array([self.control_x_min,self.control_y_min,-1]),\
                np.array([self.control_x_max,self.control_y_max,1]))

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        #Number of steps per episode
        self.step_limit = rospy.get_param("/episode_steps");

        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        #Count Steps Elapsed
        self.cumulated_steps = 0.0
        
        # Machine and Neighbor Name(Optional: Needed only for multi-agent)
        machine_name = '/machine_'+str(self.robotID);rotors_machine_name = '/firefly_'+str(self.robotID)
        neighbor_name = '/machine_'+str((self.robotID+1)%self.num_robots);rotors_neighbor_name = '/firefly_'+str((self.robotID+1)%self.num_robots)
        
        #Publish Reward
        self.reward_topic = machine_name+"/reward"
        self._reward_pub = rospy.Publisher(self.reward_topic, Float64, queue_size=1)
        self.reward_msg = Float64()

        #If Triangulating Root using Least Squares: Debug Topic
        self.triangulated_root_topic = machine_name+"/triangulated_root"
        self.triangulated_pub = rospy.Publisher(self.triangulated_root_topic, PoseArray, queue_size=1)
        self.triangulated_root = PoseArray()


            
        '''Evaluation Publishers'''
        self.centering_pub = rospy.Publisher('/centering', PointStamped, queue_size=1)
        #Publish error visualization as Covariance Ellipsoid
        self.triangulated_cov_pub=[]
        for it in range(14):
            self.triangulated_cov_pub.append(rospy.Publisher(machine_name+"/triangulated_root_cov_"+str(it), PoseWithCovarianceStamped, queue_size=1))

        #Publish error visualization as Covariance Ellipsoid
        self.triangulated_cov_pub_weighted=[]
        for it in range(14):
            self.triangulated_cov_pub_weighted.append(rospy.Publisher(machine_name+"/triangulated_weighted_root_cov_"+str(it), PoseWithCovarianceStamped, queue_size=1))


    def _check_all_systems_ready(self):
        """ Wait for actor to respawn and move """
        # rospy.logwarn("Wait for actor to respawn and move")
        # rospy.sleep(5)


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start of an episode :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0
        self._MAV_out_ws = False
        self._episode_done = False

    def _set_action(self, action):
        """
        Provide way-point for low-level controller
        """
        rospy.logdebug("Start Set Action ==>"+str(action))
        if self.ACTION_TYPE == "discrete":
            control = []
            if action == 0: #FORWARD
                control.append(self.control_x_max)
                control.append(0)
                control.append(0)
                self.last_action = "FORWARD"

            elif action == 1: #BACKWARD
                control.append(self.control_x_min)
                control.append(0)
                control.append(0)
                self.last_action = "BACKWARD"

            elif action == 2: #UP
                control.append(0)
                control.append(0)
                control.append(self.control_z_min)
                self.last_action = "UP"

            elif action == 3: #DOWN
                control.append(0)
                control.append(0)
                control.append(self.control_z_max)
                self.last_action = "DOWN"

            elif action == 4: #UP
                control.append(0)
                control.append(self.control_y_min)
                control.append(0)
                self.last_action = "LEFT"

            elif action == 5: #DOWN
                control.append(0)
                control.append(self.control_y_max)
                control.append(0)
                self.last_action = "RIGHT"

            elif action == 6: #DOWN
                control.append(0)
                control.append(0)
                control.append(0)
                self.last_action = "STOP"
        else:
            control = action

        # We tell MAV the action to execute
        self.move_base(control, 1, update_rate=100)


    def _get_obs(self):
        """
        Read robot state
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # Pose of the MAV
        pose = self.get_gt_odom()
        stamp = pose.header.stamp
        # Velocity of the MAV
        vel = self.get_velocity(stamp)
        # Position of the Target
        target = self.get_gt_target(stamp)
        
        ###################################################################################################
        '''
        Inputs to Value Network
        '''
        ###################################################################################################
        '''Translation of Actor Joints w.r.t world'''
        gt = []
        for k in range(len(gt_joints)):
            try:
                (trans,rot) = self.listener.lookupTransform( 'world_ENU',gt_joints_names[k], stamp)
                gt.extend([trans[0],trans[1],trans[2]])
                
            except:
                (trans,rot) = self.listener.lookupTransform( 'world_ENU',gt_joints_names[k], rospy.Time(0))
                gt.extend([trans[0],trans[1],trans[2]])
        
        '''Actor Root Orientation in World frame: Input to Value Function '''
        try:
            (trans,rot) = self.listener.lookupTransform('world_ENU','actor::Head', stamp)
        except:
            (trans,rot) = self.listener.lookupTransform('world_ENU','actor::Head', rospy.Time(0))
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        target_orient = y
        
        
        '''MAV Orientation in World frame: Input to Value Function '''
        try:
            (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_machine_name+'/base_link', stamp)
            self.yaw1rot_prev = rot
        except:
            (trans,rot) = self.listener.lookupTransform( 'world_ENU',self.rotors_machine_name+'/base_link', rospy.Time(0))
            print('Unable to find yaw1')
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        yaw1 = y
        
        ###################################################################################################        
        ###################################################################################################
        '''
        Inputs for Policy Network
        '''
        ################################################################################################### 
        
        '''Actor Root in Mav base link frame: Only Translation. Also used for calculating MAV bearing'''
        try:
            (trans,rot) = self.listener.lookupTransform(self.rotors_machine_name+'/base_link','actor::actor_pose', stamp)
            self.target1_prev = trans
            self.target1rot_prev = rot
        except:
            try:
                trans=self.target1_prev
                rot=self.target1rot_prev
            except:
                (trans,rot) = self.listener.lookupTransform(self.rotors_machine_name+'/base_link','actor::actor_pose', rospy.Time(0))
            print('Unable to find target:'+self.rotors_machine_name)
        theta1 = np.arctan2(trans[1],trans[0])
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        
        delta_target1 = PoseWithCovarianceStamped();
        delta_target1.pose.pose.position.x = trans[0];delta_target1.pose.pose.position.y = trans[1];delta_target1.pose.pose.position.z = trans[2]

        '''Actor Root Orientation in Mav base link frame: Only Rotation '''
        try:
            (trans,rot) = self.listener.lookupTransform(self.rotors_machine_name+'/base_link','actor::Head', stamp)
        except:
            (trans,rot) = self.listener.lookupTransform(self.rotors_machine_name+'/base_link','actor::Head', rospy.Time(0))
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        delta_orient = y        
        
        
        '''Calculate relative velocity vector in the MAV base link frame'''
        homogeneous_matrix = tf.transformations.euler_matrix(0,0,yaw1,axes='sxyz')
        homogeneous_matrix[0:3,3] = [0,0,0]       
        relative_vel = np.dot(np.linalg.inv(homogeneous_matrix),
                              np.array([vel.twist.twist.linear.y - target.twist.twist.linear.x, 
                                        vel.twist.twist.linear.x - target.twist.twist.linear.y,\
                                       -vel.twist.twist.linear.z - target.twist.twist.linear.z,1]))
        
        #Dummy zero inputs to value network during testing. 
        value_input = [0]*len(list(gt)+[pose.position.x,pose.position.y,pose.position.z,target.twist.twist.linear.x,target.twist.twist.linear.y,target.twist.twist.linear.z,vel.twist.twist.linear.x,vel.twist.twist.linear.y,-vel.twist.twist.linear.z,np.cos(yaw1), np.sin(yaw1),np.cos(target_orient), np.sin(target_orient)])
        
        scale = 2
        # Policy Inputs + Value Inputs
        observations =  [scale*delta_target1.pose.pose.position.x, scale*delta_target1.pose.pose.position.y, scale*delta_target1.pose.pose.position.z,\
                        scale*relative_vel[0],scale*relative_vel[1],scale*relative_vel[2],np.cos(theta1), np.sin(theta1),np.cos(delta_orient), np.sin(delta_orient)]+value_input


        rospy.loginfo("Observations==>"+str(observations))
        rospy.loginfo("END Get Observation ==>")

        return observations


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("Episode DONE==>")
        else:
            if self.cumulated_steps == self.step_limit:
                rospy.logerr("Episode Step Limit Reached")
                self._episode_done = True

            pose = self.get_gt_odom()

            # We see if we are outside the workspace
            if pose.position.x <= self.work_space_max and pose.position.x > self.work_space_min:
                self._MAV_out_ws = False
                if pose.position.y <= self.work_space_max and pose.position.y > self.work_space_min:
                    self._MAV_out_ws = False
                    if pose.position.z <= -2 and pose.position.z > -15:
                        rospy.logdebug("MAV Position is OK ==>["+str(pose.position.x)+","+str(pose.position.y)+","+str(pose.position.z)+"]")
                        self._MAV_out_ws = False
                    else:
                        rospy.logerr("MAV too Far in Z Pos ==>"+str(pose.position.z))
                        self._MAV_out_ws =  True
                else:
                    rospy.logerr("MAV too Far in Y Pos ==>"+str(pose.position.y))
                    self._MAV_out_ws = True
            else:
                rospy.logerr("MAV too Far in X Pos ==>"+str(pose.position.x))
                self._MAV_out_ws = True
        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        error = 0
        try:
            if not done:
                    
                ''' Distance to Target'''
                #[flag,distance] = self.is_in_desired_distance(observations, epsilon=0.1)
                
                '''Perception centering reward'''
                [flag,distance_to_imgcenter] = self.is_image_centered(observations, epsilon=0.1)
                #[flag,distance_to_imgcenter] = self.is_image_centered_alpha(observations, epsilon=0.1)
                #if distance > 5 and distance <10:
                    #reward+= (1-np.tanh(0.01*distance_to_imgcenter))
                #else:
                    #reward+= 0.1*(1-np.tanh(0.01*distance_to_imgcenter))

                #[flag,visible_joints] = self.is_person_visible(observations, epsilon=14)
                #reward+= 0.4*(visible_joints/14)

                error = self.get_hmr_error()
                error = self.get_weighted_hmr_error()
                #reward+=(1-np.tanh(error))

                rospy.logerr("@@@@  fullbody-average 3-D error: " + str(error))


                if abs(reward - 1) < 0.5:
                        rospy.logerr("")
                        rospy.logerr("@@@@@@@@@@@@@@@@@@@@@@@@ AWESOME AGENT @@@@@@@@@@@@@@@@@@@@")
                        rospy.logerr("")
                rospy.logwarn("reward=" + str(reward))
                rospy.logdebug("reward=" + str(reward))
                self.reward_msg.data = error
                self._reward_pub.publish(self.reward_msg)
                self.cumulated_reward += reward
                rospy.logwarn("Cumulated_reward=" + str(self.cumulated_reward))
                self.cumulated_steps += 1
                rospy.logwarn("Cumulated_steps=" + str(self.cumulated_steps))
        except:
            pass
        return reward

    def uncertainty_cost(self,observations, epsilon = -2, lower_bound = -4):
        target = self.get_target()
        reward = -1*(target.pose.covariance[0]+target.pose.covariance[7]+target.pose.covariance[14])
        if reward >= epsilon:
            self.good_behavior+=1
            rospy.logwarn("################### LOW TRACKING UNCERTAINTY: GOOD JOB ###########################")
            self.lost_target_flag = False
        if reward < lower_bound:
            self.lost_target_flag = True
            reward = lower_bound
        return reward

    def is_in_desired_rad_distance(self,observations,  epsilon_up=7, epsilon_down=3):
        """
        It returns True if the current position is similar to the desired position
        """
        target = self.get_gt_target()
        pose = self.get_gt_odom()
        is_in_desired_pos = False
        current_position = Point()
        current_position.x = target.pose.pose.position.x - pose.position.x
        current_position.y = target.pose.pose.position.y - pose.position.y
        current_position.z = target.pose.pose.position.z - pose.position.z

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        distance_from_target = self.get_distance_from_point(null_point,current_position)
        if distance_from_target <=epsilon_up and distance_from_target > epsilon_down:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FOLLOWING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_from_target]


    def is_in_desired_distance(self,observations,epsilon=0.5):
        """
        It returns True if the current position is similar to the desired position
        """
        target = self.get_gt_target()
        pose = self.get_gt_odom()
        is_in_desired_pos = False
        current_position = Point()
        current_position.x = target.pose.pose.position.x - pose.position.x
        current_position.y = target.pose.pose.position.y - pose.position.y
        current_position.z = 0

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        distance_from_target = self.get_distance_from_point(null_point,current_position)
        # if distance_from_target <=epsilon_up and distance_from_target > epsilon_down:
        if abs(distance_from_target-self.desired_distance) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FOLLOWING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_from_target]

    def is_in_MAV_desired_altitude(self,observations,epsilon=0.5):
        """
        It returns True if the current position is above a threshold height
        """
        pose = self.get_gt_odom()
        height = abs(pose.position.z)
        is_in_desired_pos = False
        if abs(height-self.desired_distance) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### ALTITUTDE: GOOD JOB ###########################")

        return [is_in_desired_pos, height]



    def is_in_desired_velocity(self,observations, epsilon=0.5):
        """
        It returns True if the current velocity is similar to the desired velocity
        """
        is_in_desired_vel = False
        vel = self.get_velocity()
        target = self.get_gt_target()
        current_vel = Point()
        current_vel.x = vel.twist.twist.linear.y - target.twist.twist.linear.x
        current_vel.y = vel.twist.twist.linear.x - target.twist.twist.linear.y
        current_vel.z = 0#((-vel.twist.twist.linear.z) - target_vel.twist.linear.z)
        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;
        err = self.get_distance_from_point(current_vel,null_point)
        if err<epsilon:
            is_in_desired_vel = True
            self.good_behavior+=1
            rospy.logwarn("################### VELOCITY: GOOD JOB ###########################")
        return [is_in_desired_vel, err]

    def is_image_centered(self,observations, epsilon=0.3):
        """
        It returns True if the current position is similar to the desired position
        """
        noisy_joints = self.get_noisy_joints()
        try:
            joints = np.array(noisy_joints.res).reshape((14,2))
            self.joints_prev_center = joints
        except:
            try:
                joints = self.joints_prev_center
            except:
                return [0,100000]

        is_in_desired_pos = False
        bb_center = Point()
        bb_center.x = joints[0,0]
        bb_center.y = joints[0,1]
        bb_center.z = 0.0

        img_center = Point()
        img_center.x = self.image_width/2
        img_center.y = self.image_height/2
        img_center.z = 0.0

        distance_to_imgcenter = self.get_distance_from_point(bb_center,img_center)
        distance = PointStamped()
        distance.header.stamp = rospy.Time.now()
        distance.point.x = distance_to_imgcenter
        
        if distance_to_imgcenter<= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### PERCEIVING: GOOD JOB ###########################")
        
        self.centering_pub.publish(distance)
        return [is_in_desired_pos,distance_to_imgcenter]

    def is_image_centered_alpha(self,observations, epsilon=0.3):
        """
        It returns True if the current position is similar to the desired position
        """
        bbox = self.get_alphapose_bbox()        
        
        is_in_desired_pos = False
        bb_center = Point()
        bb_center.x = (bbox.bbox[0]+bbox.bbox[2])/2
        bb_center.y = (bbox.bbox[1]+bbox.bbox[3])/2
        bb_center.z = 0.0

        img_center = Point()
        img_center.x = self.image_width/2
        img_center.y = self.image_height/2
        img_center.z = 0.0

        distance_to_imgcenter = self.get_distance_from_point(bb_center,img_center)

        if distance_to_imgcenter<= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### PERCEIVING: GOOD JOB ###########################")

        return [is_in_desired_pos,distance_to_imgcenter]

    def is_person_visible(self,observations,epsilon=14):
            noisy_joints = self.get_noisy_joints()
            is_in_desired_pos = False
            try:
                joints = np.array(noisy_joints.res).reshape((14,2))
                self.joints_prev_center = joints                       
                sum_xy = joints[:,0]+joints[:,1]
                visible = np.count_nonzero(sum_xy)                
            except:
                joints = self.joints_prev_center
                visible = 0
                print('It is Likely That Person is Not Visible')

            if visible == epsilon:
                rospy.logwarn("################### JOINT VISIBILITY: GOOD JOB ###########################")
                is_in_desired_pos = True
            return [is_in_desired_pos,visible]


    def is_desired_height(self,observations,epsilon=0.03):
        """
        It returns True if the tracking person height ratio w.r.t image height is close to a desired value
        """

        is_in_desired_pos = False
        height_ratio = observations[2]
        if abs(height_ratio-self.desired_height_ratio) <= epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### HEIGHT_RATIO: GOOD JOB ###########################")
        return [is_in_desired_pos,height_ratio]


    def is_flow_zero(self,observations, epsilon=0.1):
        """
        It returns True if the current position is similar to the desired position
        """
        is_in_desired_pos = False
        flow = Point()
        flow.x = observations[3]
        flow.y = observations[4]
        flow.z = 0.0

        null_point = Point();null_point.x = 0;null_point.y = 0;null_point.z = 0;

        person_flow = self.get_distance_from_point(flow,null_point)

        if person_flow < epsilon:
            is_in_desired_pos = True
            self.good_behavior+=1
            rospy.logwarn("################### FLOW: GOOD JOB ###########################")

        return [is_in_desired_pos,person_flow]

    def joint_detections_reward(self,observations, epsilon=0.5):
        # sum_obs = sum(observations[0:34])/34 #17 human joints
        obsx = [];obsy=[]

        sum_obsx = 0;sum_obsy=0
        std_obsx = 0;std_obsy = 0;
        for k in range(0,34,2):
            sum_obsx+=observations[k]
            sum_obsy+=observations[k+1]
        mean_obsx = sum_obsx/17;mean_obsy = sum_obsy/17;
        for k in range(0,34,2):
            std_obsx += abs(observations[k]-mean_obsx)
            std_obsy += abs(observations[k+1]-mean_obsy)
        sum_obs  = std_obsx+std_obsy
        is_in_desired_pos = False
        # sum_obs = sum(abs(np.array(observations[0:34]).reshape(17,2)-np.array(observations[0:2])))
        if sum_obs >=epsilon:
            self.good_behavior+=1
            is_in_desired_pos = True
            rospy.logwarn("################### JOINTS: GOOD JOB ###########################")

        return [is_in_desired_pos,sum_obs]


    def is_in_desired_position(self):
        """
        It return True if the tracking of the person is consistent for a given number of time steps
        """
        is_in_desired_pos = False

        if self.good_behavior>=self.happiness_threshold:
            rospy.logwarn("################### HAPPY AGENT ###########################")
            is_in_desired_pos = True

        return is_in_desired_pos


    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def is_in_desired_multiagent_distance(self,observations,epsilon=2):
        """
        It returns True if the current position is similar to the desired position
        """
        is_in_desired_pos = False
        relative_pos = Point()
        pose = self.get_gt_odom()
        pose_neighbor = self.get_neighbor_gt_odom()
        relative_pos.x = pose.position.x - pose_neighbor.position.x
        relative_pos.y = pose.position.y - pose_neighbor.position.y
        relative_pos.z = 0

        null_point = Point()
        null_point.x = 0.0
        null_point.y = 0.0
        null_point.z = 0.0

        ia_distance = self.get_distance_from_point(relative_pos,null_point)

        if ia_distance >= epsilon :
            is_in_desired_pos = True
        return [is_in_desired_pos,ia_distance]

    def get_hmr_error(self):
        hmr_joints = self.get_hmr_joints()
        stamp = hmr_joints.header.stamp
        null_point = Point()
        null_point.x = 0.0;null_point.y = 0.0;null_point.z = 0.0
        error = 0
        for k in range(len(gt_joints)):
            try:
                (trans,rot) = self.listener.lookupTransform( 'world',gt_joints_names[k], stamp)                                
                diff = Point()
                diff.x = trans[0]-hmr_joints.poses[k].position.x
                diff.y = trans[1]-hmr_joints.poses[k].position.y
                diff.z = trans[2]-hmr_joints.poses[k].position.z                
                error += (self.get_distance_from_point(diff,null_point))
            except:
                diff = Point()
                diff.x = 10;diff.y=10;diff.z=10
                error = 10
            err_cov  = PoseWithCovarianceStamped()
            err_cov.pose.pose.position = hmr_joints.poses[k].position
            err_cov.pose.covariance[0] = diff.x**2
            err_cov.pose.covariance[7] = diff.y**2
            err_cov.pose.covariance[14] = diff.z**2
            err_cov.header.stamp = stamp
            err_cov.header.frame_id = 'world'                
            self.triangulated_cov_pub[k].publish(err_cov)            
        error = error/len(gt_joints)
        return error        

    def get_weighted_hmr_error(self):
        hmr_joints = self.get_hmr_joints()
        stamp = hmr_joints.header.stamp
        null_point = Point()
        null_point.x = 0.0;null_point.y = 0.0;null_point.z = 0.0
        error = 0
        for k in range(len(gt_joints)):
            try:
                try:
                    (trans,rot) = self.listener.lookupTransform( 'world',gt_joints_names[k], stamp)
                except:
                    (trans,rot) = self.listener.lookupTransform( 'world',gt_joints_names[k], rospy.Time(0))                   
                diff = Point()
                if np.array([hmr_joints.poses[k].position.x,hmr_joints.poses[k].position.y,hmr_joints.poses[k].position.z]).all()== 0:
                        diff.x = 10;diff.y=10;diff.z=10
                        error += 10
                else:
                    diff.x = trans[0]-hmr_joints.poses[k].position.x
                    diff.y = trans[1]-hmr_joints.poses[k].position.y
                    diff.z = trans[2]-hmr_joints.poses[k].position.z                
                    error += weight[k]*(self.get_distance_from_point(diff,null_point))
            except:
                diff.x = 10;diff.y=10;diff.z=10
                error += 10
            err_cov  = PoseWithCovarianceStamped()
            err_cov.pose.pose.position = hmr_joints.poses[k].position
            err_cov.pose.covariance[0] = diff.x**2
            err_cov.pose.covariance[7] = diff.y**2
            err_cov.pose.covariance[14] = diff.z**2
            err_cov.header.stamp = stamp
            err_cov.header.frame_id = 'world'                
            self.triangulated_cov_pub_weighted[k].publish(err_cov)            
        return error
            

    def get_lsq_triangulation_error_with_noisy_gt(self,observation,epsilon=0.5):

        noisy_joints = self.get_noisy_joints()
        stamp = noisy_joints.header.stamp
        noisy_joints_neighbor = self.get_noisy_joints_neighbor(stamp)
        try:
            self.joints = np.array(noisy_joints.res).reshape((14,2))
            self.joints_prev = self.joints
        except:
            self.joints = self.joints_prev
            print('Unable to find noisy joints')
        try:
            self.joints_neighbor = np.array(noisy_joints_neighbor.res).reshape((14,2))
            self.joints_neighbor_prev = self.joints_neighbor
        except:
            self.joints_neighbor = self.joints_neighbor_prev
            print('Unable to find noisy joints from neighbor')
        # Get the camera intrinsics of both cameras
        P1 = self.get_cam_intrinsic()
        P2 = self.get_neighbor_cam_intrinsic()

        # Convert world coordinates to local camera coordinates for both cameras
        # We get the camera extrinsics as follows
        try:
            trans,rot = self.listener.lookupTransform(self.rotors_machine_name+'/xtion_rgb_optical_frame','world', stamp) #target to source frame
            self.trans_prev = trans
            self.rot_prev = rot
        except:
            trans = self.trans_prev
            rot = self.rot_prev
            print('Robot rotation unavailable')
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H1 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H1[0:3,3] = trans

        try:
            trans,rot = self.listener.lookupTransform(self.rotors_neighbor_name+'/xtion_rgb_optical_frame','world', stamp) #target to source frame
            self.trans2_prev = trans
            self.rot2_prev = rot
        except:
            trans = self.trans2_prev
            rot = self.rot2_prev
            print('Robot neighbor rotation unavailable')
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H2 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H2[0:3,3] = trans


        #Concatenate Intrinsic Matrices
        intrinsic = np.array((P1[0:3,0:3],P2[0:3,0:3]))
        #Concatenate Extrinsic Matrices
        extrinsic = np.array((H1,H2))
        joints_gt = self.get_joints_gt()
        error = 0
        self.triangulated_root = PoseArray()
        for k in range(len(gt_joints)):

            p2d1 = self.joints[k,:]
            p2d2 = self.joints_neighbor[k,:]
            points_2d = np.array((p2d1,p2d2))

            # Solve system of equations using least squares to estimate person position in robot 1 camera frame
            estimate_root  = self.lstsq_triangulation(intrinsic,extrinsic,points_2d,2)
            es_root_cam = Point();es_root_cam.x = estimate_root[0];es_root_cam.y = estimate_root[1];es_root_cam.z = estimate_root[2]

            err_cov  = PoseWithCovarianceStamped()
            #if v>4: #because head, eyes and ears lead to high error for the actor
            # if v==5 or v == 6:
            p = Pose()
            p.position = es_root_cam
            self.triangulated_root.poses.append(p)
            gt = joints_gt.poses[gt_joints[gt_joints_names[k]]].position
            error += (self.get_distance_from_point(gt,es_root_cam))

            err_cov.pose.pose.position = es_root_cam
            err_cov.pose.covariance[0] = (gt.x - es_root_cam.x)**2
            err_cov.pose.covariance[7] = (gt.y - es_root_cam.y)**2
            err_cov.pose.covariance[14] = (gt.z - es_root_cam.z)**2
            err_cov.header.stamp = rospy.Time()
            err_cov.header.frame_id = 'world'
            self.triangulated_cov_pub[k].publish(err_cov)

        # Publish all estimates
        self.triangulated_root.header.stamp = rospy.Time()
        self.triangulated_root.header.frame_id = 'world'
        self.triangulated_pub.publish(self.triangulated_root)



        is_in_desired_pos = False
        error = error/len(gt_joints)
        if error<=epsilon:
            # error = 0.4*error/epsilon
            is_in_desired_pos = True
        # else:
        #     error = 0.4

        return [is_in_desired_pos,error]

    def get_lsq_triangulation_error(self,observation,epsilon=0.5):
        # Get the camera intrinsics of both cameras
        P1 = self.get_cam_intrinsic()
        P2 = self.get_neighbor_cam_intrinsic()

        # Convert world coordinates to local camera coordinates for both cameras
        # We get the camera extrinsics as follows
        trans,rot = self.listener.lookupTransform(self.rotors_machine_name+'/xtion_rgb_optical_frame','world', rospy.Time(0)) #target to source frame
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H1 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H1[0:3,3] = trans
        print(str(self.env_id)+' '+str(self.rotors_machine_name)+':'+str(trans))
        trans,rot = self.listener.lookupTransform(self.rotors_neighbor_name+'/xtion_rgb_optical_frame','world', rospy.Time(0)) #target to source frame
        (r,p,y) = tf.transformations.euler_from_quaternion(rot)
        H2 = tf.transformations.euler_matrix(r,p,y,axes='sxyz')
        H2[0:3,3] = trans
        print(str(self.env_id)+' '+str(self.rotors_neighbor_name)+':'+str(trans))

        #Concatenate Intrinsic Matrices
        intrinsic = np.array((P1[0:3,0:3],P2[0:3,0:3]))
        #Concatenate Extrinsic Matrices
        extrinsic = np.array((H1,H2))
        joints_gt = self.get_joints_gt()
        error = 0

        self.triangulated_root = PoseArray()
        for k,v in dict_joints.items():

            p2d1 = self.joints[dict_joints[k],0:2]
            p2d2 = self.joints_neighbor[dict_joints[k],0:2]
            points_2d = np.array((p2d1,p2d2))

            # Solve system of equations using least squares to estimate person position in robot 1 camera frame
            estimate_root  = self.lstsq_triangulation(intrinsic,extrinsic,points_2d,2)
            es_root_cam = Point();es_root_cam.x = estimate_root[0];es_root_cam.y = estimate_root[1];es_root_cam.z = estimate_root[2]

            err_cov  = PoseWithCovarianceStamped()
            joints = 0
            if v>4: #because head, eyes and ears lead to high error for the actor
            # if v==5 or v == 6:
                p = Pose()
                p.position = es_root_cam
                self.triangulated_root.poses.append(p)
                gt = joints_gt.poses[alpha_to_gt_joints[k]].position
                error += (self.get_distance_from_point(gt,es_root_cam))

                err_cov.pose.pose.position = es_root_cam
                err_cov.pose.covariance[0] = (gt.x - es_root_cam.x)**2
                err_cov.pose.covariance[7] = (gt.y - es_root_cam.y)**2
                err_cov.pose.covariance[14] = (gt.z - es_root_cam.z)**2
                err_cov.header.stamp = rospy.Time()
                err_cov.header.frame_id = 'world'
                self.triangulated_cov_pub[alpha_to_gt_joints[k]].publish(err_cov)
                joints+=1

        # Publish all estimates
        self.triangulated_root.header.stamp = rospy.Time()
        self.triangulated_root.header.frame_id = 'world'
        self.triangulated_pub.publish(self.triangulated_root)



        is_in_desired_pos = False
        error = error/joints
        if error<=epsilon:
            # error = 0.4*error/epsilon
            is_in_desired_pos = True
        # else:
        #     error = 0.4

        return [is_in_desired_pos,error]


    def lstsq_triangulation(self,intrinsic,extrinsic,points_2d,cams):
        '''
        Least Squares Triangulation
        '''

        a=[]
        b=[]
        norm_points = []
        w = []

        for cam in range(cams):
            extr = extrinsic[cam,:3,:]
            # convert to homogeneous coordinates
            point = np.append(points_2d[cam,:], 1)

            # normalize the 2D points using the intrinsic parameters
            norm_points.append(np.matmul(np.linalg.inv(intrinsic[cam,:,:]), point))
            # we'll use equation 14.42 of the CV book by Simon Prince.
            # form of the equation is Ax=b.

            # generate the matrix A and vector b
            a.append(np.outer((norm_points[-1])[0:2], extr[2, 0:-1]) - extr[0:2, 0:-1])
            b.append(extr[0:-1, -1] - extr[-1, -1] * (norm_points[-1])[0:2])
            # print((norm_points[-1])[0:2])

        A = np.concatenate(a)
        B = np.concatenate(b)

        # solve with least square estimate
        x = np.linalg.lstsq(A, B, rcond=None)[0]

        return x
