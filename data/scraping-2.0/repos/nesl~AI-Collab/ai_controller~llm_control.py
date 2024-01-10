import numpy as np
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from gym_collab.envs.action import Action
import torch
import os
import re
import json
import pdb
from cnl import MessagePattern
from movement import Movement
from enum import Enum
import time
from collections import deque


class LLMControl:

    def __init__(self,openai, env, device, robotState):
        self.action_retry = 0
        self.next_loc = []
        self.item_list = []
        self.item_list_dup = []
        self.action_sequence = 0
        self.top_action_sequence = 0
        self.llm_messages = []
        self.openai = openai
        self.held_objects = []
        self.env = env
        self.sample_action_space = env.action_space.sample()
        self.device = device
        self.action_function = ""

        self.explore_location = []
        self.previous_message = []
        self.message_text = ""
        self.action_history = deque(maxlen=5)
        self.message_info = [False,"",0]
        self.ask_info_time_limit = 10
        
        self.movement = Movement(env)

        self.history_prompt = "Imagine you are a robot. You can move around a place, pick up objects and use a sensor to determine whether an object is dangerous or not. Your task is to find all dangerous objects in a room and bring them to the middle of that room. The size of the room is   by  meters. There are other robots like you present in the room with whom you are supposed to collaborate. Objects have a weight and whenever you want to pick up an object, you need to make sure your strength value is equal or greater than that weight value at any given moment. That means that whenever you carry a heavy object other robots will need to be next to you until you drop it. You start with a strength of 1, and each other robot that is next to you inside a radius of 3 meters will increase your strength by 1. If you pick up an object you cannot pick up another object until you drop the one you are carrying. Each sensor measurement you make to a particular object has a confidence level, thus you are never totally sure whether the object you are scanning is benign or dangerous. You need to compare measurements with other robots to reduce uncertainty. You can only sense objects by moving within a radius of 1 meter around the object and activating the sensor. You can sense multiple objects each time you activate your sensor, sensing all objects within a radius of 1 meter. You can exchange text messages with other robots, although you need to be at most 5 meters away from them to receive their messages and send them messages. All locations are given as (x,y) coodinates. The functions you can use are the following:\ngo_to_location(x,y): Moves robot to a location specified by x,y coordinates. Returns nothing.\nsend_message(text): Broadcasts message text. Returns nothing.\nactivate_sensor(): Activates sensor. You need to be at most 1 meter away from an object to be able to sense it. Returns a list of lists, each of the sublists with the following format: [“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”]. For example: [[“1”,”4,5”,”1”,”benign”,”0.5”],[“2”,”6,7”,”1”,”dangerous”,”0.4”]].\npick_up(object_id): Picks up an object with object id object_id. You need to be 0.5 meters from the object to be able to pick it up. Returns nothing.\ndrop(): Drops any object previously picked up. Returns nothing.\nscan_area(): Returns the locations of all objects and robots in the scene.\n"



        #self.system_prompt = "Imagine you are a robot. You can move around a place, use a sensor to determine whether an object is dangerous or benign, and pick up objects if necessary. Your task is to find all dangerous objects in the scene, pick them up and carry them to the goal area. Just remember that if you pick up an object you cannot pick up another object until you drop the one you are carrying, and you can only sense objects by getting close to the object and activating the sensor. This action will update the information relevant to such object, you should only try sensing an object once as there is no extra information obtained from doing it more times. Location is represented with (x,y) coordinates."
        self.system_prompt = "Imagine you are a robot working as part of a team. You can move around a place, use a sensor to determine whether an object is dangerous or benign, and pick up objects if necessary. Your task is to find all dangerous objects in the scene, pick them up and carry them to the goal area. Some objects are too heavy for you to carry alone, thus you need to ask for help to a specific number of teammates. The number of teammates needed is equal to the weight of the object you want to carry. Asking teammates to help you prevents them from working individually towards the common objective, so you should make responsible use of them. You can only sense objects by getting close to the desired object and activating the sensor. This action will update the information relevant to such object and add information about its status and measurement confidence. Because your sensor is faulty, it will provide a measurement with only a certain percentage of confidence, it is up to you to decide whether you trust it or not. You can exchange information about objects automatically with other teammates by getting close to them. Location is represented with (x,y) coordinates." # Whenever you think there is nothing else left to do, you can always choose to end participation."

        if self.openai:
            self.llm_messages.extend([
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": "",
                },
                {
                    "role": "function",
                    "content": "",
                },
                
                ]
            ) 
        else:
            self.llm_messages.append(
                {
                    "role": "system",
                    "content": self.history_prompt,
                }
            ) 

        self.llm_functions = [
            {
                "name": "go_to_location",
                "description": "Move to the specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "integer",
                            "description": "The ID of the object to move next to. After you have picked up an object, you can call this function with an object_id of -1 to move to the goal area.", # If you want to move to the middle of the room at location (10,10), use -1.",
                        },
                        
                    },
                    "required": ["object_id"],
                },
            },
            {
                "name": "activate_sensor",
                "description": "Activates sensor and returns information about nearby objects, such as if they are dangerous or benign. ",
                "parameters":{ "type": "object", "properties": {}},
                
            },
            {
                "name": "pick_up",
                "description": "Picks up a nearby object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "integer",
                            "description": "The ID of the object to pick up. You need to be maximum 1 coordinate unit away of it. ",
                        }
                    },
                    "required": ["object_id"],
                },
            },
            {
                "name": "drop",
                "description": "Drops any object previously picked up.",
                "parameters":{ "type": "object", "properties": {}},
                
            },


        ]
	
        self.setup_llm(self.device)
        log_file = "log_ai.txt"
        self.log_f = open(log_file,"w")
        
        
        self.other_agents = [self.Other_Agent() for r in range(env.action_space["robot"].n-1)]
        self.message_text = ""
        self.chosen_object_idx = -1
        self.target_location = []
        self.movement = Movement(env)
        self.action_index = self.State.llm_state
        self.previous_next_loc = []
        self.nearby_other_agents = []
        self.help_requests = []
        self.object_of_interest = ""
        self.help_time_limit2 = 30
       
    class Other_Agent:
        
        def __init__(self):
            self.my_location = {"ego_location": [], "goal_location": [], "next_location": []} #Location of agent being controlled here
            self.my_carrying = False
            self.my_team = ""
            
            self.other_location = {"ego_location": [], "goal_location": [], "next_location": []} 
            self.team = ""
            self.carrying = False
            self.items = {}
            self.assignment = "None"
            self.observations = deque(maxlen=5)
            
    class State(Enum):
        llm_state = 0
        drop_object = 1
       
    def message_processing(self,received_messages, robotState, info):
    
        time_log = info["time"]
        
        objects_str = {}
    
        for rm in received_messages:
            
            print("Received message:", rm)
            template_match = False
            
            agent_idx = info['robot_key_to_index'][rm[0]]
            
            
            if MessagePattern.carry_help_accept(self.env.robot_id) in rm[1]:
            
                template_match = True
                
                object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(self.chosen_object_idx)]
                
                obs_string = "Offered me help to carry object " + str(object_id)
                
                self.other_agents[agent_idx].observations.append(obs_string)
                    
                return_value,self.message_text,_ = self.movement.message_processing_carry_help_accept(rm, {"weight": robotState.items[self.chosen_object_idx]["item_weight"], "index": self.chosen_object_idx}, self.message_text)
                
                
                if return_value == 1:
                    
                    if not (robotState.items[self.chosen_object_idx]['item_location'][0] == -1 and robotState.items[self.chosen_object_idx]['item_location'][1] == -1):
                        self.target_location = robotState.items[self.chosen_object_idx]['item_location']
                        self.object_of_interest = object_id
                        #self.target_object_idx = self.heavy_objects["index"][self.chosen_heavy_object]
                        
                        self.action_index = self.State.llm_state
                    else: #Somehow we end here
                        self.message_text += MessagePattern.carry_help_reject(rm[0])
                        



            if re.search(MessagePattern.carry_help_regex(),rm[1]):
            
                rematch = re.search(MessagePattern.carry_help_regex(),rm[1])
                
                template_match = True
                
                self.other_agents[agent_idx].observations.append("Asked me to help carry object " + rematch.group(2))

                """                  
                if re.search(MessagePattern.carry_help_regex(),message_text): #This means the robot is preparing to ask for help and reject the help request, we shouldn't allow this
                    message_text = message_text.replace(re.search(MessagePattern.carry_help_regex(),message_text).group(), "")
                    self.movement.asked_help = False
                    self.movement.asked_time = time.time()
                    action_index = self.last_action_index
                """
                
                if not robotState.object_held and not self.movement.helping and not self.movement.being_helped and not self.movement.accepted_help and not self.movement.asked_help: # accept help request
                    #message_text += MessagePattern.carry_help_accept(rm[0])
                    #self.movement.accepted_help = rm[0]
                    self.help_requests.append(rm[0])

                    #self.helping = rm[0]
                    #self.action_index = self.State.check_neighbors
                    
                else: #reject help request
                    self.message_text += MessagePattern.carry_help_participant_reject(rm[0])
                    print("Cannot help", not robotState.object_held, not self.movement.helping, not self.movement.being_helped, not self.movement.accepted_help, not self.movement.asked_help)
            
            
                """
                template_match = True
            
                self.message_text,self.action_index,_ = self.movement.message_processing_carry_help(rm, robotState, self.action_index, self.message_text)
                """
                
            if re.search(MessagePattern.follow_regex(),rm[1]) or MessagePattern.carry_help_cancel() in rm[1] or MessagePattern.carry_help_reject(self.env.robot_id) in rm[1] or MessagePattern.carry_help_finish() in rm[1] or MessagePattern.carry_help_complain() in rm[1]:
            
                template_match = True
            
                self.action_index,_ = self.movement.message_processing_help(rm, self.action_index, self.State.llm_state)
                
                    
                if re.search(MessagePattern.follow_regex(),rm[1]):
                    rematch = re.search(MessagePattern.follow_regex(),rm[1])
                
                    if rematch.group(1) == str(self.env.robot_id):
                        self.other_agents[agent_idx].observations.append("Asked me to follow him")
                    else:
                        self.other_agents[agent_idx].observations.append("Asked " + rematch.group(1) + " to follow him")
                        
                elif MessagePattern.carry_help_cancel() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Cancelled his request for help")
                    
                elif MessagePattern.carry_help_finish() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Finished moving heavy object with help from others")
                
                elif MessagePattern.carry_help_complain() in rm[1]:
                    self.other_agents[agent_idx].observations.append("Dismissed his team for not collaborating effectively")
                    
                    
            if re.search(MessagePattern.carry_help_reject_regex(),rm[1]):
                
                template_match = True
                
                rematch = re.search(MessagePattern.carry_help_reject_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Rejected my offer to help him")
                else:
                    self.other_agents[agent_idx].observations.append("Rejected " + rematch.group(1) + "'s offer to help him")    
                      
            
            if re.search(MessagePattern.location_regex(),rm[1]):
            
                template_match = True
                
                carrying_variable = self.other_agents[agent_idx].carrying
                team_variable = self.other_agents[agent_idx].team
                
                if not (self.movement.being_helped and rm[0] in self.movement.being_helped and self.action_index == self.State.drop_object):            
                    self.message_text,self.action_index,_ = self.movement.message_processing_location(rm, robotState, info, self.other_agents, self.target_location, self.action_index, self.message_text, self.State.llm_state, self.next_loc)
                    
                
                rematch = re.search(MessagePattern.location_regex(),rm[1])
                
                obs_string = ""
                
                if not self.other_agents[agent_idx].carrying and self.other_agents[agent_idx].carrying != carrying_variable:
                    obs_string += "Dropped an object"
                    
                    
                
                if not self.other_agents[agent_idx].team and self.other_agents[agent_idx].team != team_variable:
                    
                    if obs_string:
                        obs_string += ", "
                
                    if team_variable != str(self.env.robot_id):
                        obs_string += "Stopped helping agent " + team_variable
                    else:
                        obs_string += "Helped me carry an object"
                            
                
                
                if rematch.group(1) != "location":
                
                    if obs_string:
                        obs_string += ", "
                
                    obs_string += "Announced his current objective is " + rematch.group(1)
                    
                if rematch.group(5):
                    if obs_string:
                        obs_string += ", "
                        
                    obs_string += "Carried object " + rematch.group(6)
                    
                    
                if rematch.group(7):
                    
                        
                    if rematch.group(8) != str(self.env.robot_id):
                        if obs_string:
                            obs_string += ", "
                            
                        obs_string += "Is helping agent " + rematch.group(8)

                
                if obs_string and obs_string not in self.other_agents[agent_idx].observations:
                    self.other_agents[agent_idx].observations.append(obs_string)
            
            if MessagePattern.wait(self.env.robot_id) in rm[1] or re.search(MessagePattern.move_order_regex(),rm[1]):
                template_match = True
                self.target_location, self.action_index, _ = self.movement.message_processing_wait(rm, info, self.target_location, self.action_index)
                self.object_of_interest = ""
                
            if re.search(MessagePattern.wait_regex(),rm[1]):
                template_match = True
                rematch = re.search(MessagePattern.wait_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Waited for me to pass")
                else:
                    self.other_agents[agent_idx].observations.append("Waited for " + rematch.group(1) + " to pass")
                
            if MessagePattern.move_request(self.env.robot_id) in rm[1]:
                template_match = True
                
                
                self.message_text,self.action_index,_ = self.movement.message_processing_move_request(rm, robotState, info, self.action_index, self.message_text)
                
            if re.search(MessagePattern.move_request_regex(),rm[1]):
                template_match = True
                rematch = re.search(MessagePattern.move_request_regex(),rm[1])
                
                if rematch.group(1) == str(self.env.robot_id):
                    self.other_agents[agent_idx].observations.append("Asked me to move")
                else:
                    self.other_agents[agent_idx].observations.append("Asked " + rematch.group(1) + " to move")
                    
            if re.search(MessagePattern.sensing_help_regex(),rm[1]): #"What do you know about object " in rm[1]:
                rematch = re.search(MessagePattern.sensing_help_regex(),rm[1])
                
                template_match = True
                
                object_id = rematch.group(1) #rm[1].strip().split()[-1] 
                object_idx = info['object_key_to_index'][object_id]
                
                self.other_agents[agent_idx].observations.append("Asked me for information about object " + str(object_id))
                
                self.message_text += MessagePattern.item(robotState,object_idx,object_id, info, self.env.robot_id, self.env.convert_to_real_coordinates)
                
                if not self.message_text:
                     self.message_text += MessagePattern.sensing_help_negative_response(object_id)
            if re.search(MessagePattern.item_regex_full(),rm[1]) or re.search(MessagePattern.item_regex_full_alt(),rm[1]):
            
                template_match = True
                
                new_rm = list(rm)
                new_rm[1] += MessagePattern.translate_item_message(new_rm[1],self.env.robot_id)
            
                obs_str = "Shared information with me of objects: ["
            
                if rm[1] not in objects_str:
                    objects_str[rm[1]] = []
                
            
                for ridx,rematch in enumerate(re.finditer(MessagePattern.item_regex_full(),new_rm[1])):
                
                    object_id = rematch.group(1)
                
                    if object_id == str(self.message_info[1]):
                        self.message_info[0] = True
                        
                    MessagePattern.parse_sensing_message(rematch, new_rm, robotState, info, self.other_agents, self.env.convert_to_grid_coordinates)
                    
                    
                        
                    if object_id not in objects_str[rm[1]]:
                        if objects_str[rm[1]]:
                            obs_str += ", "
                        obs_str += object_id
                        objects_str[rm[1]].append(object_id)
                    
                obs_str += "]"
                self.other_agents[agent_idx].observations.append(obs_str)
                    
            if re.search(MessagePattern.sensing_help_negative_response_regex(),rm[1]):
            
                template_match = True
            
                rematch = re.search(MessagePattern.sensing_help_negative_response_regex(),rm[1])
                
                self.other_agents[agent_idx].observations.append("Told me he doesn't have any information about object " + rematch.group(1))
                
                if rematch.group(1) == str(self.message_info[1]):
                    self.message_info[0] = True


            if not template_match:
                pass
   
    def get_neighboring_agents(self, robotState, ego_location):
    
        nearby_other_agents = []
        #Get number of neighboring robots at communication range
        for n_idx in range(len(robotState.robots)):
            if "neighbor_location" in robotState.robots[n_idx] and not (robotState.robots[n_idx]["neighbor_location"][0] == -1 and robotState.robots[n_idx]["neighbor_location"][1] == -1) and self.env.compute_real_distance([robotState.robots[n_idx]["neighbor_location"][0],robotState.robots[n_idx]["neighbor_location"][1]],[ego_location[0][0],ego_location[1][0]]) < self.env.map_config['communication_distance_limit']:
                nearby_other_agents.append(n_idx)
                
        return nearby_other_agents    
        
    def modify_occMap(self,robotState, occMap, ego_location, info):
    
        self.movement.modify_occMap(robotState, occMap, ego_location, info, self.next_loc)
        
        if self.action_index != self.State.drop_object and robotState.object_held:
            for agent_id in self.movement.being_helped: #if you are being helped, ignore locations of your teammates

                agent_idx = info['robot_key_to_index'][agent_id]
                other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                
                if not (other_robot_location[0] == -1 and other_robot_location[1] == -1) and occMap[other_robot_location[0],other_robot_location[1]] != 5:
                    occMap[other_robot_location[0],other_robot_location[1]] = 3

    def control(self,messages, robotState, info, next_observation):
        #print("Messages", messages)
        
        terminated = False
        
        self.occMap = np.copy(robotState.latest_map)
        
        ego_location = np.where(self.occMap == 5)
        
        self.modify_occMap(robotState, self.occMap, ego_location, info)
        
        self.nearby_other_agents = self.get_neighboring_agents(robotState, ego_location)
        
        
        if messages: #Process received messages
            self.message_processing(messages, robotState, info)
        
        self.message_text += MessagePattern.exchange_sensing_info(robotState, info, self.nearby_other_agents, self.other_agents, self.env.robot_id, self.env.convert_to_real_coordinates) #Exchange info about objects sensing measurements
        
        if not self.message_text:
        
            if self.action_index == self.State.llm_state or self.action_index == self.State.drop_object:
                if not self.action_function or self.help_requests:
                    history_prompt,function_str = self.ask_llm(messages, robotState, info, [], self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                    
                        print("Starting...")
                    
                        #self.action_function = input("Next action > ").strip()
                        #self.action_function = "scan_area()"
                        self.action_function = "self." + function_str[:-1]
                
                        #if not ("drop" in self.action_function or "activate_sensor" in self.action_function or "scan_area" in self.action_function):
                        if not ("explore" in self.action_function):
                            self.action_function += ","
                        
                        self.action_function += "robotState, next_observation, info)"
                    else:
                        self.action_function = ""

                action, action_finished,function_output = eval(self.action_function)

                if action_finished:
                    self.action_sequence = 0
                    self.top_action_sequence = 0
                    history_prompt,function_str = self.ask_llm(messages, robotState, info, function_output, self.nearby_other_agents, self.help_requests)
                    
                    if function_str:
                    
                        self.action_function = "self." + function_str[:-1]
                
                        #if not ("drop" in self.action_function or "activate_sensor" in self.action_function or "scan_area" in self.action_function):
                        if not ("explore" in self.action_function):
                            self.action_function += ","
                        
                        self.action_function += "robotState, next_observation, info)"
                    else: #No function selected
                        self.action_function = ""
            else:
                action = self.sample_action_space
                action["action"] = -1
                action["num_cells_move"] = 1
            
                previous_action_index = self.action_index
                
                self.message_text,self.action_index,self.target_location,self.next_loc, low_action = self.movement.movement_state_machine(self.occMap, info, robotState, self.action_index, self.message_text, self.target_location,self.State.llm_state, self.next_loc, ego_location, -1)
                self.object_of_interest = ""
                
                if previous_action_index == self.movement.State.wait_message and not self.movement.asked_help:
                    self.action_function = ""
                
                action["action"] = low_action
                
            
            if self.nearby_other_agents: #If there are nearby robots, announce next location and goal

                self.message_text, self.next_loc = self.movement.send_state_info(action, self.next_loc, self.target_location, self.message_text, self.other_agents, self.nearby_other_agents, ego_location, robotState, self.object_of_interest, self.held_objects)    
                
            if self.message_text: #Send message first before doing action
                

                if re.search(MessagePattern.location_regex(),self.message_text):
                    rematch = re.search(MessagePattern.location_regex(),self.message_text)
                    target_goal = eval(rematch.group(2))
                    target_loc = eval(rematch.group(3))
                    
                    if target_goal != target_loc and not (self.previous_message and self.previous_message[0] == target_goal and self.previous_message[1] == target_loc): #Only if there was a change of location do we prioritize this message

                        self.previous_message = [target_goal,target_loc]

                        
                        action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
                        print("SENDING MESSAGE", info['time'], self.message_text)
                        self.message_text = ""

                        
                
        else:
        
            action,_,_ = self.send_message(self.message_text, robotState, next_observation, info)
            print("SENDING MESSAGE2", info['time'], self.message_text)
            self.message_text = ""
            

        

        if action["action"] == -1 or action["action"] == "":
            
            action["action"] = Action.get_occupancy_map.value
            print("STUCK")

        print("action index:",self.action_index, "action:", Action(action["action"]), ego_location, self.action_function)
        
        if "end_participation" in self.action_function:
            terminated = True
        

        return action,terminated
        
    """
    def compute_real_distance(self,neighbor_location,ego_location):
    
        res = np.linalg.norm(np.array([neighbor_location[0],neighbor_location[1]]) - np.array([ego_location[0],ego_location[1]]))*self.env.map_config['cell_size']
        
        return res
    """
    
    
    '''   
    def go_to_location(self,x,y, robotState, next_observation, info):
                
                
        ego_location = np.where(robotState.latest_map == 5)
        
        finished = False
        action = self.sample_action_space
        action["action"] = -1
        action["num_cells_move"] = 1
        
        output = []
        
        """
        if action_sequence == 0:
            action_sequence += 1
            action = Action.get_occupancy_map.value
        """
        if self.action_sequence == 0:
            self.path_to_follow = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
            
            if not self.path_to_follow:
                action["action"] = Action.get_occupancy_map.value
                finished = True
                output = -1
            else:
            
                next_location = [ego_location[0][0],ego_location[1][0]]
                action["action"] = self.movement.position_to_action(next_location,self.path_to_follow[0],False)
            
                previous_action = ""
                repetition = 1
                action["num_cells_move"] = repetition 
                
                """
                previous_action = ""
                repetition = 1
                next_location = [ego_location[0][0],ego_location[1][0]]
                for p_idx in range(len(path_to_follow)):
                    action["action"] = position_to_action(next_location,path_to_follow[p_idx],False)
                    
                    if not p_idx:
                        previous_action = action["action"]
                        next_location = path_to_follow[p_idx]
                    else:
                        if previous_action == action["action"]:
                            repetition += 1
                            next_location = path_to_follow[p_idx]
                        else:
                            break
                            
                for r in range(repetition-1):
                    path_to_follow.pop(0)
                    
                action["num_cells_move"] = repetition   
                """ 
                self.action_sequence += 1
                #print(path_to_follow, ego_location)

        else:
            if any(next_observation['action_status'][:2]):
                if ego_location[0][0] == self.path_to_follow[0][0] and ego_location[1][0] == self.path_to_follow[0][1]:
                    if self.path_to_follow:
                        self.path_to_follow.pop(0)
                        
       
                if self.path_to_follow:    
                    next_location = [ego_location[0][0],ego_location[1][0]]
                    action["action"] = self.movement.position_to_action(next_location,self.path_to_follow[0],False)
                
                    previous_action = ""
                    repetition = 1
                    action["num_cells_move"] = repetition 
                    
                    """
                    for p_idx in range(len(path_to_follow)):
                        action["action"] = position_to_action(next_location,path_to_follow[p_idx],False)
                        
                        if not p_idx:
                            previous_action = action["action"]
                            next_location = path_to_follow[p_idx]
                        else:
                            if previous_action == action["action"]:
                                repetition += 1
                                next_location = path_to_follow[p_idx]
                            else:
                                break
                    for r in range(repetition-1):
                        path_to_follow.pop(0)
                    action["num_cells_move"] = repetition  
                    """
                else:
                    action["action"] = Action.get_occupancy_map.value
                    finished = True
                    
        return action,finished,output
    '''    
    
    def sense_object(self, object_id, robotState, next_observation, info):
        
        finished = False
        
        if self.top_action_sequence == 0:
        
            chosen_location = robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
        
            if (chosen_location[0] == -1 and chosen_location[1] == -1) or self.occMap[chosen_location[0],chosen_location[1]] != 2: #if there is no object in the correct place
                finished = True
        
            action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
        elif self.top_action_sequence == 1:
            action, finished, output = self.activate_sensor(robotState, next_observation, info)
        
    
        return action, finished, output
        
        
    def collect_object(self, object_id, robotState, next_observation, info):
    
        finished = False
        output = []
        
        action = self.sample_action_space
        
        ego_location = np.where(robotState.latest_map == 5)
        
        self.chosen_object_idx = info['object_key_to_index'][str(object_id)]
        
        if self.top_action_sequence == 0:

            chosen_location = robotState.items[info['object_key_to_index'][str(object_id)]]["item_location"]
            
            if (chosen_location[0] == -1 and chosen_location[1] == -1) or self.occMap[chosen_location[0],chosen_location[1]] != 2: #if there is no object in the correct place
                finished = True
            
            action, temp_finished, output = self.go_to_location(object_id, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
                self.movement.asked_time = time.time()
                self.movement.being_helped_locations = []
        elif self.top_action_sequence == 1:
        
            wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, [], ego_location,self.message_text)
        
            if not wait_for_others:        
                action, temp_finished, output = self.pick_up(object_id, robotState, next_observation, info)
                self.held_object = object_id
                if temp_finished:
                    self.top_action_sequence += 1
                    self.movement.being_helped_locations = []
                    self.previous_next_loc = []
   
   
                    g_coord = []
                    for g_coord in self.env.goal_coords:
                        if not self.occMap[g_coord[0],g_coord[1]]:
                            self.target_location = g_coord
                            self.object_of_interest = "goal"
                            break
                    
                    
                    
            elif not combinations_found: #No way of moving                          
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.llm_state,self.message_text,message=MessagePattern.carry_help_finish())
                finished = True
            elif time.time() - self.movement.asked_time > self.help_time_limit2:                           
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.llm_state,self.message_text,message=MessagePattern.carry_help_complain())
                finished = True
            else:
                action["action"] = Action.get_occupancy_map.value 
                
        elif self.top_action_sequence == 2:

            self.action_index = self.State.drop_object
            
            if not robotState.object_held:            
                action["action"],self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.llm_state,self.message_text,message=MessagePattern.carry_help_complain())
                
            else:
            
            
                for agent_id in self.movement.being_helped: #remove locations with teammates

                    agent_idx = info['robot_key_to_index'][agent_id]
                    other_robot_location = robotState.robots[agent_idx]["neighbor_location"]
                    
                    if not (other_robot_location[0] == -1 and other_robot_location[1] == -1):
                        self.occMap[other_robot_location[0],other_robot_location[1]] = 3
                            
                
                loop_done = False
                
                if not self.previous_next_loc or (self.previous_next_loc and self.previous_next_loc[0].tolist() == [ego_location[0][0],ego_location[1][0]]):
                    action["action"], self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index)
                    
                    
                    print("HAPPENING", action["action"], self.next_loc)
                    
                    if not action["action"] and isinstance(action["action"], list):
                        loop_done = True
                     
                    if not loop_done and self.next_loc:
                        self.previous_next_loc = [self.next_loc[0]]
                        self.movement.being_helped_locations = []
                    
                        print("PEFIOUVS",self.movement.being_helped_locations, self.next_loc, self.previous_next_loc)
                        
                    
                    
                    
                
                if not loop_done:
                    wait_for_others,combinations_found,self.message_text = self.movement.wait_for_others_func(self.occMap, info, robotState, self.nearby_other_agents, self.previous_next_loc, ego_location,self.message_text)
                    
                    if not combinations_found: #No way of moving
                        self.top_action_sequence += 1
                        action["action"] = Action.get_occupancy_map.value
                        _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.llm_state,self.message_text,message=MessagePattern.carry_help_finish())
                
                
                        
                if loop_done or not wait_for_others: #If carrying heavy objects, wait for others
                    
                    action["action"], self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(self.target_location[0],self.target_location[1], self.occMap, robotState, info, ego_location, self.action_index)
                    
                    if self.next_loc and self.previous_next_loc and not self.previous_next_loc[0].tolist() == self.next_loc[0].tolist(): #location changed
                        self.previous_next_loc = []
                        
                        
                    """    
                    if self.occMap[self.target_location[0],self.target_location[1]] == 2: #A package is now there
                        self.action_index = self.State.pickup_and_move_to_goal
                        self.movement.being_helped_locations = []
                    """
                
                    if not action["action"] and isinstance(action["action"], list): #If already next to drop location
                        action["action"] = Action.get_occupancy_map.value
                        self.top_action_sequence += 1
                        self.target_location = self.past_location
                        self.object_of_interest = ""

                    else:
                        self.past_location = [ego_location[0][0],ego_location[1][0]]
                        
                    self.movement.asked_time = time.time()
                    
                elif time.time() - self.movement.asked_time > self.help_time_limit2:
                    self.top_action_sequence += 1
                    action["action"] = Action.get_occupancy_map.value
                    _,self.message_text,self.action_index = self.movement.cancel_cooperation(self.State.get_closest_object,self.message_text,message=MessagePattern.carry_help_complain())
                    
                elif action["action"] != Action.drop_object.value:
                    action["action"] = Action.get_occupancy_map.value
                    print("waiting for others...")
            
            
            """
            action, temp_finished, output = self.go_to_location(-1, robotState, next_observation, info)
            if temp_finished:
                self.top_action_sequence += 1
                self.action_index = self.State.llm_state
            """
        elif self.top_action_sequence == 3:
            action, finished, output = self.drop(robotState, next_observation, info)
        
            if self.movement.being_helped:
                self.message_text += MessagePattern.carry_help_finish()
                self.movement.asked_time = time.time()
            self.movement.being_helped = []
            self.movement.being_helped_locations = []
        
    
        return action, finished, output
        
    def explore(self, robotState, next_observation, info):
    
        
    
        action, finished, output = self.go_to_location(-2, robotState, next_observation, info)
        
        if action["action"] == -1:
            action["action"] = Action.get_occupancy_map.value
            self.action_index = self.State.llm_state
            finished = True
        
        if finished:
            self.explore_location = []
        
        return action, finished, output
        
        
    def ask_for_help(self, object_id, robotState, next_observation ,info):
    
        action = self.sample_action_space
        action["robot"] = 0
        finished = False
        output = []
        
        
        #if self.action_sequence == 0:
        object_idx =info['object_key_to_index'][str(object_id)]
        self.message_text += MessagePattern.carry_help(str(object_id),robotState.items[object_idx]["item_weight"]-1)
        self.movement.asked_help = True
        self.movement.asked_time = time.time()
        if not self.action_index == self.movement.State.wait_free and not self.action_index == self.movement.State.wait_random:
            self.movement.last_action_index = self.action_index
        self.action_index = self.movement.State.wait_message
        self.chosen_object_idx = object_idx
        print("ASKING HELP")
        
        self.action_function = "self.collect_object(" + str(object_id) + ", robotState, next_observation, info)"
        
        #    self.action_sequence += 1
            
        #elif self.action_sequence == 1:
        #    action, finished, output = self.collect_object(object_id, robotState, next_observation, info)    
        
        
        """
        
        if self.action_sequence == 0:
            action["action"] = Action.send_message.value
    
            action["message"] = MessagePattern.carry_help(object_id,self.heavy_objects['weight'][chosen_object]-1)
            
            self.action_sequence += 1
            
        elif self.action_sequence == 1:
            action["action"] = Action.get_occupancy_map.value
            
        """
        
        
        
        return action,finished,output
    
    def go_to_location(self, object_id, robotState, next_observation, info):
                
                
        ego_location = np.where(robotState.latest_map == 5)
        
        finished = False
        action = self.sample_action_space
        action["action"] = -1
        action["num_cells_move"] = 1
        
        output = []
        
        """
        if action_sequence == 0:
            action_sequence += 1
            action = Action.get_occupancy_map.value
        """
        
        if object_id == -1: #Return to middle of the room
            x = 10
            y = 10
        elif object_id == -2: #Explore
        
            if not self.explore_location:
                still_to_explore = np.where(robotState.latest_map == -2)
                
                closest_dist = float('inf')
                closest_idx = -1
            

                for se_idx in range(len(still_to_explore[0])):
                    unknown_loc = [still_to_explore[0][se_idx],still_to_explore[1][se_idx]]
                    
                    unknown_dist = self.env.compute_real_distance(unknown_loc,[ego_location[0][0],ego_location[1][0]])
                    
                    if unknown_dist < closest_dist:
                        closest_dist = unknown_dist
                        closest_idx = se_idx
                        
                x = still_to_explore[0][closest_idx]
                y = still_to_explore[1][closest_idx]
                
                self.explore_location = [x,y]
            else:
                x = self.explore_location[0]
                y = self.explore_location[1]
            
  
        elif str(object_id).isalpha(): #Agent
            
            robot_idx = info['robot_key_to_index'][str(object_id)]
            
            if (robotState.robots[robot_idx]["neighbor_location"][0] == -1 and robotState.robots[robot_idx]["neighbor_location"][1] == -1):
                action["action"] = Action.get_occupancy_map.value
                return action,True,output
            
            x,y = robotState.robots[robot_idx]["neighbor_location"]
            
            
        else:
            try:
            
                item_idx = info['object_key_to_index'][str(object_id)]
            
                if (robotState.items[item_idx]["item_location"][0] == -1 and robotState.items[item_idx]["item_location"][1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    return action,True,output
            
                x,y = robotState.items[item_idx]["item_location"]
                
                
            except:
                pdb.set_trace()
        

        low_action, self.next_loc, self.message_text, self.action_index = self.movement.go_to_location(x, y, self.occMap, robotState, info, ego_location, self.action_index)


        """
        self.path_to_follow = self.movement.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
        
        if not self.path_to_follow or x == self.path_to_follow[0][0] and y == self.path_to_follow[0][1]:
            action["action"] = Action.get_occupancy_map.value
            finished = True
        else:
        
            next_location = [ego_location[0][0],ego_location[1][0]]
            action["action"] = self.movement.position_to_action(next_location,self.path_to_follow[0],False)
        
            previous_action = ""
            repetition = 1
            action["num_cells_move"] = repetition 
                
        """
        
        if not low_action and isinstance(low_action, list):
            action["action"] = Action.get_occupancy_map.value
            finished = True
        else:
            action["action"] = low_action
            
        
                    
        return action,finished,output
        
    def activate_sensor(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = False
        output = []


        if self.action_sequence == 0:
            self.action_sequence += 1
            action["action"] = Action.danger_sensing.value
            
        elif self.action_sequence == 1:
            self.item_list = info["last_sensed"]
            #print(item_list)
            self.item_list_dup = self.item_list.copy()
            
            if not self.item_list: #No items scanned
                action["action"] = Action.get_occupancy_map.value
                finished = True
            else:
            
                object_key = self.item_list.pop(0)
                
                action["action"] = Action.check_item.value    
                action["item"] = info["object_key_to_index"][object_key]
                
                if not self.item_list: #item list finished
                    self.action_sequence += 2
                else:
                    self.action_sequence += 1
            
        elif self.action_sequence == 2:
            object_key = self.item_list.pop(0)
            action["action"] = Action.check_item.value    
            action["item"] = info["object_key_to_index"][object_key]
            
          
            if not self.item_list:
                self.action_sequence += 1
           
                
        elif self.action_sequence == 3:
            #[“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”
            for key in self.item_list_dup:
            
                ob_idx = info["object_key_to_index"][key]
            
                if robotState.items[ob_idx]["item_danger_level"] == 1:
                    danger_level = "benign"
                else:
                    danger_level = "dangerous"
                    
                output.append([str(key),str(int(robotState.items[ob_idx]["item_location"][0]))+","+str(int(robotState.items[ob_idx]["item_location"][1])),str(robotState.items[ob_idx]["item_weight"]),danger_level,str(robotState.items[ob_idx]["item_danger_confidence"][0])])
            
            action["action"] = Action.get_occupancy_map.value
        
            finished = True
                
            
            
        """
        elif action_sequence == 1:
            if next_observation['action_status'][2]:
                action["action"] = Action.check_item.value
                action["item"] = item_number
                
                if item_number < len(robotState.items):
                    item_number += 1
                else:
                    action_sequence += 1
                    
        elif action_sequence == 2:
            most_recent = 0
            items_keys = []
            for ri_key in robotState.items:
                robotState.items[ri_key]["item_time"]
        """
        
        return action,finished,output
        
    def send_message(self,message, robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = []
        action["action"] =  Action.send_message.value
        
        action["message"] = message

        return action,finished,output
        
    def pick_up(self,object_id,robotState, next_observation, info):
        
        
        action = self.sample_action_space
        action["action"] = -1
        
        ego_location = np.where(robotState.latest_map == 5)

        output = []
        
        finished = False
        
        
        
        if self.action_sequence == 0:
        
            if not robotState.object_held:
            
                self.action_retry = 0
            
                self.action_sequence += 1    

                ob_idx = info["object_key_to_index"][str(object_id)]
             

                if not robotState.items[ob_idx]["item_weight"]:
                    output = -2
                    finished = True
                    action["action"] = Action.get_occupancy_map.value
                else:
                    location = robotState.items[ob_idx]["item_location"]
                    action["action"] = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
                    if action["action"] == -1 or (location[0] == -1 and location[1] == -1):
                        action["action"] = Action.get_occupancy_map.value
                        finished = True
                        output = -1
                
            else:
                action["action"] = Action.get_occupancy_map.value
                
                finished = True
                output = -3
            
        elif self.action_sequence == 1:
            if robotState.object_held or self.action_retry == 2:
                action["action"] = Action.get_occupancy_map.value
        
                finished = True
                
                if self.action_retry == 2 and not robotState.object_held:
                    output = -1
                else:
                    self.held_objects = str(object_id)
            else:
                ob_idx = info["object_key_to_index"][str(object_id)]
                location = robotState.items[ob_idx]["item_location"]
                action["action"] = self.movement.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
                self.action_retry += 1
                
                if action["action"] == -1 or (location[0] == -1 and location[1] == -1):
                    action["action"] = Action.get_occupancy_map.value
                    finished = True
                    output = -1
    
        
            

        return action,finished,output
             
    def drop(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = self.held_objects
        
        action["action"] = Action.drop_object.value

        return action,finished,output
        
    def end_participation(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = []

        return action,finished,output
        
    def scan_area(self,robotState, next_observation, info):
        action = self.sample_action_space

        finished = True

        output = []
        
        action["action"] = Action.get_occupancy_map.value

        return action,finished,output
        
    def help(self, agent_id, robotState, next_observation, info):
    
        action = self.sample_action_space
        
        finished = False
        
        output = []
        
        if self.action_sequence == 0:
            self.message_text += MessagePattern.carry_help_accept(agent_id)
            self.movement.accepted_help = agent_id
            
            self.action_index = self.movement.State.wait_follow
            self.action_sequence += 1 
            
        elif self.action_sequence == 1:
            finished = True
        
        action["action"] = Action.get_occupancy_map.value
        
        return action,finished,output
        
    def ask_info(self, object_id, robotState, next_observation, info):
    
        action = self.sample_action_space
        
        finished = False
        
        output = []
        
        if self.action_sequence == 0:
            self.message_text += MessagePattern.sensing_help(object_id)
            self.message_info[0] = False
            self.message_info[1] = object_id
            self.message_info[2] = time.time()
            
            self.action_sequence += 1 
            
        elif (self.action_sequence == 1 and self.message_info[0]) or time.time() - self.message_info[2] > self.ask_info_time_limit:
            finished = True
        
        action["action"] = Action.get_occupancy_map.value
        
        return action,finished,output
        
    def ask_llm(self,messages, robotState, info, output, nearby_other_agents, help_requests):


        prompt = ""
        token_output = ""
        possible_functions = ["drop","go_to_location","pick_up","send_message","activate_sensor", "scan_area"]
        last_action = ""
        action_output_prompt = ""
        possible_actions = {}
        ego_location = np.where(robotState.latest_map == 5)

        '''
        if "pick_up" in self.action_function:
            first_index = self.action_function.index("(") + 1
            arguments = self.action_function[first_index:].split(",")
            
            if output == -1:
                action_output_prompt = "Failed to pick up object " + arguments[0] + ". "
            elif output == -2:
                action_output_prompt = "You cannot pick an unknown object. " #+ arguments[0] + ". "
            elif output == -3:
                action_output_prompt = "You cannot pick up an object if you are already carrying one. "
            else:
                action_output_prompt = "Succesfully picked up object " + arguments[0] + ". "
                
            last_action = "pick_up(" + arguments[0] + ")"
                
        elif "go_to_location" in self.action_function:
            first_index = self.action_function.index("(") + 1
            arguments = self.action_function[first_index:].split(",")
            
            if output:
                action_output_prompt = "Failed to move to location (" + arguments[0] + "," + arguments[1] + "). "
            else:
                action_output_prompt = "Arrived at location "
                
                if int(arguments[0]) > -1:
                    action_output_prompt += "next to object " + arguments[0] + ". "
                else:
                    action_output_prompt += "in goal area. "
                
            last_action = "go_to_location(" + arguments[0] + ")"
            
        elif "activate_sensor" in self.action_function:
            if not output:
                action_output_prompt = "No objects are close enough to be sensed. "
            else:
                action_output_prompt = "Sensor was activated successfully, sensed "
                
                for ob_idx,ob in enumerate(output):
                    
                    if ob_idx:
                        action_output_prompt += ", "
                
                    action_output_prompt += "object " + ob[0]
                    
                action_output_prompt += ". "
                
            last_action = "activate_sensor()"    
            """
                prompt += "The sensing results are the following: " + str(output) + ". "
            else:
                prompt += "No objects are close enough to be sensed. "
            """
        elif "send_message" in self.action_function:
            action_output_prompt = "Sent message. "
            last_action = "send_message()" 
        elif "drop" in self.action_function:
            if output:
                action_output_prompt = "Dropped object " + output[0] + ". "
                self.held_objects.pop(0)
            else:
                action_output_prompt = "No object to drop. "
                
            last_action = "drop()" 

        '''    
            
        #objects_location = np.where(robotState.latest_map == 2)
    
        """
        if objects_location[0].size > 0:
            prompt += "Last seen objects at locations: "
        """
        prompt += "Objects discovered so far: ["
        
        for ob_idx,ob in enumerate(robotState.items):
        
            if ob_idx:
                prompt += ", "
        
            object_id = list(info['object_key_to_index'].keys())[list(info['object_key_to_index'].values()).index(ob_idx)]
            
            #prompt += "{'object_id': " + object_id + ", 'location': (" + str(ob["item_location"][0]) + "," + str(ob['item_location'][1]) + ')'
            
            if ob["item_location"][0] == -1 and ob["item_location"][1] == -1:
                distance = "inf"
            else:
                _, next_locs, _, _ = self.movement.go_to_location(ob["item_location"][0], ob["item_location"][1], self.occMap, robotState, info, ego_location, self.action_index, checking=True)
                distance = len(next_locs)
                
                if not distance:
                    distance = "inf"
            
            prompt += "{'object_id': " + object_id + ", 'distance': " + str(distance)
            #prompt += "Object " + object_id + " at location (" + str(ob["item_location"][0]) + "," + str(ob['item_location'][1]) + ')'
            
            if ob['item_weight']:
                prompt += ", weight: " + str(ob["item_weight"])
            
            
            if ob['item_danger_level']:
                prompt += ", 'status': "
                #prompt += " is "
                if ob['item_danger_level'] == 1:
                    prompt += "'benign'"
                else:
                    prompt += "'dangerous'"
                    
                prompt += ", 'confidence': " + str(round(ob["item_danger_confidence"][0]*100,2)) + "%"
                
                if not (ob["item_location"][0] == -1 and ob["item_location"][1] == -1) and tuple(ob["item_location"]) not in self.env.goal_coords:
                    
                    if len(nearby_other_agents) + 1 >= ob['item_weight'] and distance != "inf":
                        if ob['item_weight'] == 1:
                            possible_actions["Collect object " + object_id] = "collect_object(" + object_id + ")"
                        elif not help_requests:
                            possible_actions["Ask for help to collect object " + object_id] = "ask_for_help(" + object_id + ")" 
                    
            else:
                if not (ob["item_location"][0] == -1 and ob["item_location"][1] == -1) and tuple(ob["item_location"]) not in self.env.goal_coords and distance != "inf":
                    possible_actions["Sense object " + object_id] = "sense_object(" + object_id + ")"
                
            """    
                prompt += ", 'already_sensed': true"
            else:
                prompt += ", 'already_sensed': false"
                
            if tuple(ob["item_location"]) in self.env.goal_coords:
                prompt += ", 'inside_goal_area': true"
            else:
                prompt += ", 'inside_goal_area': false"
            """
                
            prompt += "}"
            
            
            if nearby_other_agents and tuple(ob["item_location"]) not in self.env.goal_coords:
                possible_actions["Ask information about object " + object_id] = "ask_info(" + object_id + ")"
            
        prompt += "]. "
        
        """
        for ob_idx in range(len(objects_location[0])):
            
                
            map_key = str(objects_location[0][ob_idx]) + '_' + str(objects_location[1][ob_idx])
            for m_idx,map_object in enumerate(robotState.map_metadata[map_key]):
            
                if ob_idx or m_idx:
                    prompt +=", "
            

                object_id = map_object[0]

                    
                prompt += "{object_id: " + object_id + ", location: (" + str(objects_location[0][ob_idx]) + "," + str(objects_location[1][ob_idx]) + ")}"
            
        prompt += "]. "
        """
        
        """
        robots_location = np.where(robotState.latest_map == 3)
        
        '''
        if robots_location[0].size > 0:
            prompt += "Last seen robots at locations: "
        '''
        for ob_idx in range(len(robots_location[0])):
        
            map_key = str(robots_location[0][ob_idx]) + '_' + str(robots_location[1][ob_idx])
            for m_idx,map_object in enumerate(robotState.map_metadata[map_key]):
            
                if ob_idx or m_idx:
                    prompt +=", "
            

                object_id = map_object           
          
                prompt += "Robot " + object_id + " is at location (" + str(robots_location[0][ob_idx]) + "," + str(robots_location[1][ob_idx]) + ")"
            
        prompt += ". "
            
        """
        
        prompt += "Agents in your team: ["
        
        for a_idx, ag in enumerate(robotState.robots):
        
            if a_idx:
                prompt += ", "
            
            agent_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(a_idx)]
        
            #prompt += "{'agent_id': " + agent_id + ", 'location': (" + str(ag["neighbor_location"][0]) + "," + str(ag['neighbor_location'][1]) + ')'
            
            if ag["neighbor_location"][0] == -1 and ag["neighbor_location"][1] == -1:
                distance = "inf"
            else:
                _, next_locs, _, _ = self.movement.go_to_location(ag["neighbor_location"][0], ag["neighbor_location"][1], self.occMap, robotState, info, ego_location, self.action_index, checking=True)
                distance = len(next_locs)
                
                if not distance:
                    distance = "inf"
            
            prompt += "{'agent_id': " + agent_id + ", 'distance': " + str(distance)

            if self.other_agents[a_idx].observations:
            
                prompt += ", 'observations': ["

                for obs_idx,obs_a in enumerate(self.other_agents[a_idx].observations):
                    if obs_idx:
                        prompt += ", "
                    prompt += obs_a
                     
                prompt += "]"

            prompt += "}"
            
            if a_idx not in nearby_other_agents and not (ag["neighbor_location"][0] == -1 and ag["neighbor_location"][1] == -1):
                possible_actions["Go with agent " + agent_id] = "go_to_location('" + agent_id + "')"
            
        
           
            
        prompt += "]. "
        
        #possible_actions["End participation"] = "end_participation()"
        
        action_status_prompt = prompt
        
        if self.openai and action_status_prompt:
            self.llm_messages[1]["content"] = action_status_prompt
        
        status_prompt = ""
        
        """
        if not robotState.object_held and self.held_objects:
            status_prompt += "Accidentally dropped object " + self.held_objects[0] + ". "
            self.held_objects.pop(0)
        if robotState.object_held and self.held_objects:
            status_prompt += "You are carrying object " + self.held_objects[0] + ". "    
        
            
        if messages:

            self.process_messages(messages)
            

            status_prompt += "You have received the following messages: "
            
            for m in messages:
                status_prompt += "Message from " + str(m[0]) + ": '" + m[1] + "'. "
            status_prompt += ". "
        """
        
        
        #status_prompt += "Your current strength is " + str(robotState.strength) + ". "
        
        ego_location = np.where(robotState.latest_map == 5)
        
        #status_prompt += "Your current location is " + "(" + str(ego_location[0][0]) + "," + str(ego_location[1][0]) + "). "
        
        unexplored = np.where(robotState.latest_map == -2)
        
        if unexplored[0].size:
            explore_str = "Explore"
            possible_actions[explore_str] = "explore()"
        
        explored = round((robotState.latest_map.size-unexplored[0].size)/robotState.latest_map.size*100,2)
        
        status_prompt += "Percentage of the world explored is " + str(explored) + "%. "
        
        if help_requests:
            for _ in range(len(help_requests)):
                hr = help_requests.pop(0)
                #status_prompt += hr + " is requesting help to carry an object. "
                possible_actions["Help " + hr + " to carry an object"] = "help('" + hr + "')"
                
        
        """
        if (ego_location[0][0],ego_location[1][0]) in self.env.goal_coords:
            status_prompt += "You are located in the goal area. "
        """
        
        """
        if self.action_history:
            status_prompt += "Action history: " + self.action_history + ". "
        if last_action:
            status_prompt += "Last action: " + last_action + ". "
            status_prompt += "Last action output: " + action_output_prompt
            
            if self.action_history:
                self.action_history += ", "
            
            self.action_history += last_action

        """
        
        if self.action_history:
            status_prompt += "Action history: ["
            for ah_index,ah in enumerate(self.action_history):
                if ah_index:
                    status_prompt += ", "
                status_prompt += ah
                
            status_prompt += "]. "
            
            
        
        prompt += status_prompt
        
        
        team_str = ""
        for agent_idx in range(len(self.other_agents)):
            
            if self.other_agents[agent_idx].team == self.env.robot_id:
                agent_id = list(info['robot_key_to_index'].keys())[list(info['robot_key_to_index'].values()).index(agent_idx)]
                
                if not team_str:
                    team_str += "Agents currently helping me: " + agent_id
                else:
                    team_str += ", " + agent_id
                    
        prompt += team_str
                
        
        #prompt += "What will be your next action? Write a single function call."
        
        prompt += "Output only a number indicating the action to perform from the next list of possible actions and then give a short explanation:"
        #prompt += "Output only a number indicating the action to perform from the next list of possible actions, do not output anything else:"
        
        
        
        
        
        for p_idx,p in enumerate(possible_actions.keys()):
            prompt += "\n" + str(p_idx+1) + ") " + p
        
        prompt += "\n"
       
        
        print("Starting to call LLM")
        
        final_prompt = ""
        
        if not self.openai:

        
            #print(prompt)
            
            
            
            for m in self.llm_messages:
                if m["role"] == "system":
                    final_prompt += m["content"]
                elif m["role"] == "user":
                    final_prompt += "USER >> " + m["content"]
                elif m["role"] == "assistant":
                    final_prompt += "ASSISTANT >> " + m["content"]
                final_prompt += "\n"
                
            final_prompt += "USER >> " + prompt + "\nASSISTANT >>\n"
            
            print(final_prompt)
            
            temperature = 0.7
            num_beams = 1
            while True:
            
                inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
                
                generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=100, num_beams=num_beams, early_stopping=True, temperature=temperature) #, generation_config=config)

                token_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                new_tokens = inputs.input_ids.size()[1]
                
                function_output = self.tokenizer.batch_decode(generate_ids[:,new_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                function_output = function_output.replace('\\', '')
                print(function_output)
	            
                x_str = re.search(r"[a-z_]+\([^)]*\)",function_output)
	            
                
                if x_str:
                

                    function_output = x_str.group()
                    
                    
                    if function_output and any(func_str in function_output for func_str in possible_functions):
                    
                        if "send_message" in function_output:
                            parent_index = function_output.index('(')
                            second_parent_index = function_output.index(')')
                            msg = function_output[parent_index+1:second_parent_index].replace('"','\\"').replace("'","\\'")
                            function_output = "send_message('" + msg +  "')"
                  
                        self.llm_messages.append({"role":"user","content":action_status_prompt + status_prompt})
                        self.llm_messages.append({"role":"assistant","content":function_output})
                        #output = history_prompt + action_status_prompt + status_prompt + "\nROBOT >> " + function_output #final_prompt + function_output
                        break
                    else:
                        num_beams += 1
                        print("Increasing beams to", num_beams)
                        #final_prompt = output + "\nENVIRONMENT >> That is not a function, please try again.\nROBOT >>\n"
                else:
                    num_beams += 1
                    print("Increasing beams to", num_beams)
                    #final_prompt = output + "\nENVIRONMENT >> That is not a function, please try again.\nROBOT >>\n"
	            
                #pdb.set_trace()
                
        else:


            response = openai.ChatCompletion.create(
              model= "gpt-4", #"gpt-4", #"gpt-3.5-turbo",
              messages=[
                    self.llm_messages[0],
                    {"role": "user", "content": prompt}
                ],
              #functions=self.llm_functions,
              #function_call="auto"
            )
            print(self.llm_messages[0], {"role": "user", "content": prompt})
            print(response)

            #pdb.set_trace()
            response_message = response["choices"][0]["message"]
            
            message_content = response_message.get("content")
            
            function_output = ""
            function_index = -1
            
            for p in possible_actions.keys():
                if p in message_content:
                    function_output = possible_actions[p]
                    function_index = p
                    break
                    
            if not function_output: #If given number only
                rematch = re.search("\d+", message_content)
                if rematch:
                    possible_functions = list(possible_actions.keys())
                    func_idx = int(rematch.group())-1
                    if func_idx < len(possible_functions):
                        function_index = possible_functions[func_idx]
                        function_output = possible_actions[function_index]
                        
                    
            if function_output:
                self.action_history.append(function_index)        
            
            """
            if response_message.get("function_call"):
                #self.llm_messages.append({"role": "user", "content": status_prompt})
                
                function_name = response_message["function_call"]["name"]
                function_args = json.loads(response_message["function_call"]["arguments"])
                
                function_output = function_name + "("
                for key_idx,key in enumerate(function_args.keys()):
                    if key_idx:
                        function_output += ","
                        
                        
                    if "send_message" in function_output:
                        function_output += "'" + str(function_args[key]).replace('"','\\"').replace("'","\\'") +  "'"
                    else:
                        function_output += str(function_args[key])
                    
                function_output += ")"
                
                '''
                self.llm_messages.append({"role": "function", "name": function_name, "content": ""})
                
                for msg in self.llm_messages:
                    final_prompt += msg["role"] + " >> "
                    if "name" in msg and msg["content"]:
                        final_prompt += msg["name"] + " > "
                        
                    final_prompt += msg["content"] + '\n'
                '''

            """        
        
        #self.log_f.write(final_prompt + function_output + '\n')
        
        return token_output+'\n', function_output
            
    def setup_llm(self,device):


        
        if self.openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            pretrained_model = "eachadea/vicuna-13b-1.1" #"tiiuae/falcon-40b-instruct" #"eachadea/vicuna-13b-1.1"
            
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16, device_map='sequential', max_memory={0: '12GiB', 1: '20GiB'}, revision='main', low_cpu_mem_usage=True, offload_folder='offload')
            
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            
        
        
