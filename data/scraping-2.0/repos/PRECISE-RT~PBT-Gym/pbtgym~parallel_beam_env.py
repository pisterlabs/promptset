# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:02:14 2023

@author: Rob Chambers

This file contains the ParallelBeamEnv class, inheriting from openAI gym's Env
class. It's purpose is to create a basic model of a proton beam dosing a patient
so that reinforcing learning techniques. 

Adding user input step functions for use with any user defined dosing model.
"""

from .requirements import *
from .beam import Beam
from .dose_model import complex_dose, bit_wise_dose
from .brain_region import BrainRegion

# Requires CANCER, EXTERNAL, LEARN_SPACE global variables at present. 
# This is an issue from notebook -> package conversion; hopefully soon to be fixed
class ParallelBeamEnv(Env):
    def __init__(self):
        super(ParallelBeamEnv, self).__init__()
        self.cancer = CANCER
        self.external = EXTERNAL
        self.learn_space = LEARN_SPACE
        self.x_min = self.learn_space[0]
        self.x_max = self.learn_space[1]
        self.y_min = self.learn_space[2]
        self.y_max = self.learn_space[3]
        self.search_space = self.bound_search_space(self.cancer)
        self.beam = Beam(self.search_space) 
        self.action_space = spaces.MultiDiscrete([3,3,3,2])# Position of spot, angle of beam, beam intensity
        self.x_size = self.x_max - self.x_min
        self.y_size = self.y_max - self.y_min
        self.observation_shape = (self.y_size, self.x_size)
        self.target_dose = 0.7 # Dose prescription measured in Gy
        self.canvas = np.zeros((512,512), dtype=np.float32)
        self.target_mask = self.target_dose * np.array(self.cancer.boolean_map[self.y_min:self.y_max, self.x_min:self.x_max], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low = np.array([0,0]), 
                                    high = np.array(self.observation_shape),
                                    dtype=int),
            "dose distribution": spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype=np.float32),
            "target mask": spaces.Box(low = np.zeros(self.observation_shape),
                                        high = np.ones(self.observation_shape),
                                        dtype=np.float32)})
        self.state = {
            "position": [0, 0],
            "dose distribution": self.canvas[self.y_min:self.y_max, self.x_min:self.x_max],
            "target mask": self.target_mask}
        self.treatment_period = 1024 # Number of steps per learning perdiod
        self.theta_resolution = 45 # How many degrees changed per step, this has to be 45 for use with our current dose model
        self.default_dose = 0.1 # Default amount to be used until variable dosing is implemented. Can also be a reference value for decrete choice dosing e.g choices are 0, 0.25, 0.5, 1, 1.5, 2, 2.5x reference dose when choosing
        self.step_size = 3
        self.reward_stack = self.calculate_reward_stack()
        self.reward_map = self.reward_stack[0]
        self.update_reward_map()
        self.alpha = 0.0005
        self.beta = 0.0005
        self.step_without_dose = 0

    def set_patient(self, cancer, external):
        """ Changes the patient that the environment is using.

        Args:
            cancer (BrainRegion): the brain region object for the cancer of the new patient
            external (BrainRegion): the brain region object for the external of the new patient
        """
        self.cancer = cancer
        self.external = external
        self.reset()
    
    def set_learn_space(self, min_x, max_x, min_y, max_y):
        """Sets the bounds for the learn space - the size of the space returned in the step function

        Args:
            min_x (double): minimum x coordinate
            max_x (double): maximum x coordinate
            min_y (double): minimum y coordinate
            max_y (double): maximum y coordinate
        """
    def step(self, action):
        """
        Applies the action chosen by the Agent. Calculates 'reward'. Updates the
        'state'. Decreases the treatment period by 1. Updates the 'done' status.
        Returns 'info'. 
        """
        # Initially set reward to zero so if no action taken, reward does not carry over from previous step
        reward = 0
        # Check if plan is done
        if self.treatment_period <= 0: 
            done = True
        else:
            done = False
        
        # Translational Shift
        del_yx = (action[0:2] - 1) * self.step_size # The 1 here can be changed to change step size
        self.beam.move(del_yx[0], del_yx[1])
        self.state["position"] = np.array([self.beam.y-self.y_min, self.beam.x-self.x_min])
        
        # Angular Shift
        del_theta = (action[2] - 1) * self.theta_resolution
        self.beam.rotate(del_theta)

        # Apply_dose and calculate corresponding reward. Penalise 'no dosing'.
        if action[3] == 1:
            reward = self.apply_bortfeld_dose()# Change the dosing methods here
            self.state["dose distribution"] = self.canvas[self.y_min:self.y_max, self.x_min:self.x_max]
            self.step_without_dose = 0

        else:
            self.step_without_dose += 1
            
        if self.step_without_dose > 9:# Might need tuning to allow more steps, but this is plenty for now
            reward -= 1
        
        # Evaluate state of the environment
        bias = self.alpha * np.sum(self.target_mask) + self.beta * np.sum(self.target_mask**2)
        
        reward += (bias + self.evaluate_state())
        
        # Check for overdosing
        if (np.max(self.canvas) > 0.99): # 0.99 to allow for slight dosing over the cancer target dose (0.7 currently)
            reward -= 10
            done = True
        
        # Reduce treatment period by 1; Retaining treatment period approach in case method changes in the future
        self.treatment_period -= 1

        info = {}

        return self.state, reward, done, info

    def reset(self):
        """
        Resets environment to initial state. Currently set to zeros but could also
        be set to either base CT image. This will also have to change if we move
        from 'Box' to 'Dict' space. 
        """
        # Random Slice
        slice_number = np.random.randint(0, high=len(self.cancer.r_uids))
        self.cancer.slice_number = slice_number
        self.match_uid(self.cancer, self.external)
        self.cancer.update_slices()
        self.external.update_slices()
        self.beam.update_search_space(self.cancer.contour_map)
        self.target_mask = self.target_dose * np.array(self.cancer.boolean_map[self.y_min:self.y_max, self.x_min:self.x_max], dtype=np.float32)
        self.canvas = np.zeros((512,512), dtype=np.float32)
        # If other brain region objects added as vector, can iterator through them here

        self.update_reward_map()
        self.beam.x = np.random.randint(self.beam.x_min, self.beam.x_max)
        self.beam.y = np.random.randint(self.beam.y_min, self.beam.y_max)

        self.treatment_period = 1024
        self.state = {
            "position": np.array([self.beam.y-self.y_min, self.beam.x-self.x_min], dtype=int),
            "dose distribution": self.canvas[self.y_min:self.y_max, self.x_min:self.x_max],
            "target mask": self.target_mask}
        
        return self.state
    
    def evaluate_state(self):
        # Calculate dose conformity and dose homogeneity
        conformity = -np.sum(np.abs(self.canvas[self.y_min:self.y_max, self.x_min:self.x_max] - self.target_mask))
        homogeneity = -np.sum((self.canvas[self.y_min:self.y_max, self.x_min:self.x_max]  - self.target_mask)**2)

        # Calculate the penalty for delivering dose to critical structures
        # To be inplemented when more BrainRegions are passed to the environment
        
        # Combine the metrics with weights alpha and beta, with a distance-based incentive as well.
        # Can replace '/1000' in distance metric by another weighting factor 
        reward = self.alpha * conformity + self.beta * homogeneity + (self.cancer.distance_map[self.beam.y, self.beam.x]/1000) 
                # + self.theta * critical_structure_penalty - for when critical structures are added

        return reward

    def render(self, mode = "human"):
        """
        Renders the dosing information of the AI upon a CT image.
        """
        assert mode in ["human", "rgb_array", "cancer_contour"], "Invalid mode, must be either \"human\" or \"rgb_array\" or \"cancer_contour\""
        if mode == "human":
            plt.imshow(self.canvas)
            plt.savefig("LatestDose")
            return None
    
        elif mode == "rgb_array":
            return self.canvas
        
        elif mode == "cancer_contour":
            return self.cancer.boolean_map
            

    def is_beam_over_cancer(self):
        """
        Checks if the whole beam is over the cancer contour and if the dose
        exceeds the maximum.

        NOTE: Redundant, or soon to be so.
        """
        x = self.beam.x
        y = self.beam.y        
        beam_over_cancer = self.cancer.boolean_map[(y-1):(y+2),(x-1):(x+2)]
        dose = np.max(self.canvas[(y-1):(y+2),(x-1):(x+2)])
        if (np.sum(beam_over_cancer) == 9) and (dose < 0.7): # Just a placeholder for the dose limit
            return True
        else:
            return False

    def apply_dose(self):
        """
        Applies dose (currently just a fixed amount) at given angle and position.
        Currently just going to place 3x3 dose centered on beam position in order
        to implement proof of concept. Then can increase levels of complexity
        from there.
        """
        square_dose = self.default_dose * np.ones((3,3))
        x = self.beam.x
        y = self.beam.y
        dose_overlap = np.sum(self.reward_map[(y-1):(y+2),(x-1):(x+2)] * square_dose)
        normalised_reward = dose_overlap / (9 * self.default_dose)
        self.canvas[(y-1):(y+2),(x-1):(x+2)] += square_dose
        return normalised_reward

    def apply_bortfeld_dose(self):
        """
        Creates a line of a given thickness between two points and applies a 
        custom gradient to said line (function of distance along the line).
        Then calculates a normalised reward based on the overlap of dose and
        contour.
        """
        x = self.beam.x
        y = self.beam.y
        theta = self.beam.theta        
        
        radius = 2
        n=0
        increments = np.array([])

        if theta == 0:
            # Move in a straight line in y-direction
            if self.within_patient(y+n, x):
                # Move until the beam exits the patient (apprximating 0 attentuation outside the patient)
                while(self.within_patient(y+n, x)):
                    increments = np.append(increments, n)
                    n += 1
                # Record distance (depth) into patient
                depth = np.copy(n)
                n = 0
                # Calculate values at each point along the line using the Bortfeld approximation
                values = self.calculate_bortfield_gradient(depth - increments, increments[-1])
                # Increment the canvas with each value along the line
                for value in values:
                    self.canvas[y+n,x-radius:x+radius+1] += value
                    n += 1

        elif theta == 180 & self.within_patient(y-n,x):
            while(self.within_patient(y-n,x)):
                increments = np.append(increments, n)
                n += 1
            depth = np.copy(n)
            n = 0
            values = self.calculate_bortfield_gradient(depth - increments, depth)
            for value in values:
                self.canvas[y-n,x-radius:x+radius+1] += value
                n += 1

        elif theta == 90 & self.within_patient(y,x+n):
            while(self.within_patient(y,x+n)):
                increments = np.append(increments, n)
                n += 1
            depth = np.copy(n)
            n = 0
            values = self.calculate_bortfield_gradient(increments, depth)
            for value in values:
                self.canvas[y-radius:y+radius+1,x+n] += value
                n += 1

        elif theta == 270 & self.within_patient(y,x-n):
            while(self.within_patient(y,x-n)):
                increments = np.append(increments, n)
                n += 1
            depth = np.copy(n)
            n = 0
            values = self.calculate_bortfield_gradient(depth - increments, depth)
            for value in values:
                self.canvas[y-radius:y+radius+1,x-n] += value
                n += 1

        elif theta < 90 or theta > 270:
            m = np.rint(np.tan(theta*np.pi/180))
            if self.within_patient(y+n,int(x+m*n)):
                while(self.within_patient(y+n,int(x+m*n))):
                    distance = n * np.sqrt(1 + m**2)
                    increments = np.append(increments, distance)
                    n += 1
                depth = np.copy(distance)
                n = 0
                values = self.calculate_bortfield_gradient(depth - increments, depth)
                for value in values:
                    for r in range(2*radius+1):
                        k = r - radius
                        self.canvas[y+n-k, int(x+m*(n+k))] += value
                    for r in range(2*radius):
                        k = r - radius
                        self.canvas[y+n+k, int(x+m*(n-k-1))] += value
                    n += 1

        elif theta > 90 and theta < 270:
            m = np.rint(np.tan(theta*np.pi/180))
            if self.within_patient(y-n,int(x-m*n)):
                while(self.within_patient(y-n,int(x-m*n))):
                    distance = n * np.sqrt(1 + m**2)
                    increments = np.append(increments, distance)
                    n += 1
                depth = np.copy(distance)
                n = 0
                values = self.calculate_bortfield_gradient(depth - increments, depth)
                for value in values:
                    for r in range(2*radius+1):
                        k = r - radius
                        self.canvas[y-n-k,int(x-m*(n-k))] += value
                    for r in range(2*radius):
                        k = r - radius
                        self.canvas[y-n+k, int(x-m*(n+k+1))] += value
                    n += 1

        return 0
    
    def calculate_bortfield_gradient(self, distance, range):
        # R0 = 13.5   #range
        sigma = 0.27    #range straggling sigma
        epsilon = 0.2   #low energy contamination
        p = 1.77    #exponent of range-energy relationship
        if range == 0:
            return [0]
        out = 0.05*0.65*(self.cyl_gauss(-1/p,(distance-range)/sigma)+sigma*(0.01394+epsilon/range)*self.cyl_gauss(-1/p-1,(distance-range)/sigma))
        return out
    
    def cyl_gauss(self, a, x):
        "Calculate product of Gaussian and parabolic cylinder function"
        y = np.copy(x)
        branch = -12.0   #for large negative values of the argument we run into numerical problems, need to approximate result
        x1 = x[np.where(x<branch)]
        y1 = math.sqrt(2*math.pi)/special.gamma(-a)*(-x1)**(-a-1)
        y[np.where(x<branch)] = y1

        x2 = x[np.where(x>=branch)]
        y2a = special.pbdv(a,x2)[0]     #special function yielding parabolic cylinder function, first array [0] is function itself
        y2b = np.exp(-x2*x2/4)
        y2 = y2a*y2b

        y[np.where(x>=branch)] = y2

        return y

    def bound_search_space(self, region):
        """
        Runs through each element in binary map for each slice in a region and
        outputs the maximum and minimum boundaries for each contour region.
        """
        min_x = 512
        max_x = 0
        min_y = 512
        max_y = 0
        for k in range(len(region.binary_stack)):
            for i in range(512):
                for j in range(512):
                    if region.binary_stack[k][j,i] == 1:
                        if i < min_x:
                            min_x = i
                        if i > max_x:
                            max_x = i
                        if j < min_y:
                            min_y = j
                        if j > max_y:
                            max_y = j

        return (int(min_x), int(max_x), int(min_y), int(max_y))

    def within_cancer(self, j, i):
        """
        Returns a True if the coordinates are within the contour, else returns
        False.
        """
        
        if max(j,i) > 512 or min(j,i) < 0:
            return False
        elif self.cancer.contour_map[j,i] == 1:
            return True
        else:
            return False

    def within_patient(self, j, i):
        """
        Returns a True if the coordinates are within the contour, else returns
        False.
        """
        if max(j,i) > 512 or min(j,i) < 0:
            return False
        elif self.external.contour_map[j,i] == 1:
            return True
        else:
            return False
    
    def calculate_reward_stack(self):
        """
        Calculates a 3D 
        """
        reward_arrays = []
        
        for i in range(len(self.cancer.boolean_stack)):
            self.cancer.slice_number = i
            self.match_uid(self.cancer, self.external)
            self.cancer.update_slices()
            self.external.update_slices()
            ith_reward_map = self.calculate_reward_map()
            reward_arrays.append(ith_reward_map)
        reward_stack = np.stack(reward_arrays)
        return reward_stack

    def calculate_reward_map(self):
        """
        Calculates a reward map that priritises the cancer contour.
        """
        reward_map = np.zeros((512,512))
        # Calculates lowest hierachy first
        reward_map[self.external.boolean_stack[self.external.slice_number]] = -1

        # Calculates highest hierachy last
        reward_map[self.cancer.boolean_stack[self.cancer.slice_number]] = 1

        return reward_map
    
    def update_reward_map(self):
        """
        Updates the reward map to the givn slice index.
        """
        self.reward_map = self.reward_stack[self.cancer.slice_number]

    def match_uid(self, region_1, region_2):
        """
        Takes region and a slice index of that region and finds the 
        corresponding slice in a second given region using referenced UID
        matching. If a match is found it updates the second region's slice 
        number.
        """
        slice_id = region_1.r_uids[region_1.slice_number]
        for i in range(len(region_2.r_uids)):
            if region_2.r_uids[i] == slice_id:
                region_2.slice_number = i
                return None
                
        raise Exception(f"Slice {region_1.slice_number} from {region_1.contour_name} could not be matched with {region_2.contour_name}.")
