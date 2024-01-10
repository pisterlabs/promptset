import numpy as np
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from gym_collab.envs.action import Action
import torch
import os
import re
import json
import pdb

class LLMControl:

    def __init__(self,openai, room_size, sample_action_space, device):
        self.action_retry = 0
        self.path_to_follow = []
        self.item_list = []
        self.item_list_dup = []
        self.action_sequence = 0
        self.llm_messages = []
        self.openai = openai
        self.held_objects = []
        self.sample_action_space = sample_action_space
        self.device = device

        self.history_prompt = "Imagine you are a robot. You can move around a place, pick up objects and use a sensor to determine whether an object is dangerous or not. Your task is to find all dangerous objects in a room and bring them to the middle of that room. The size of the room is " + room_size + " by " + room_size +" meters. There are other robots like you present in the room with whom you are supposed to collaborate. Objects have a weight and whenever you want to pick up an object, you need to make sure your strength value is equal or greater than that weight value at any given moment. That means that whenever you carry a heavy object other robots will need to be next to you until you drop it. You start with a strength of 1, and each other robot that is next to you inside a radius of 3 meters will increase your strength by 1. If you pick up an object you cannot pick up another object until you drop the one you are carrying. Each sensor measurement you make to a particular object has a confidence level, thus you are never totally sure whether the object you are scanning is benign or dangerous. You need to compare measurements with other robots to reduce uncertainty. You can only sense objects by moving within a radius of 1 meter around the object and activating the sensor. You can sense multiple objects each time you activate your sensor, sensing all objects within a radius of 1 meter. You can exchange text messages with other robots, although you need to be at most 5 meters away from them to receive their messages and send them messages. All locations are given as (x,y) coodinates. The functions you can use are the following:\ngo_to_location(x,y): Moves robot to a location specified by x,y coordinates. Returns nothing.\nsend_message(text): Broadcasts message text. Returns nothing.\nactivate_sensor(): Activates sensor. You need to be at most 1 meter away from an object to be able to sense it. Returns a list of lists, each of the sublists with the following format: [“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”]. For example: [[“1”,”4,5”,”1”,”benign”,”0.5”],[“2”,”6,7”,”1”,”dangerous”,”0.4”]].\npick_up(object_id): Picks up an object with object id object_id. You need to be 0.5 meters from the object to be able to pick it up. Returns nothing.\ndrop(): Drops any object previously picked up. Returns nothing.\nscan_area(): Returns the locations of all objects and robots in the scene.\n"



        self.system_prompt = "Imagine you are a robot. You can move around a place, pick up objects and use a sensor to determine whether an object is dangerous or not. Your task is to find all dangerous objects in a room and bring them to the middle of that room. The size of the room is " + room_size + " by " + room_size +" meters. There are other robots like you present in the room with whom you are supposed to collaborate. Objects have a weight and whenever you want to pick up an object, you need to make sure your strength value is equal or greater than that weight value at any given moment. That means that whenever you carry a heavy object other robots will need to be next to you until you drop it. You start with a strength of 1, and each other robot that is next to you inside a radius of 3 meters will increase your strength by 1. If you pick up an object you cannot pick up another object until you drop the one you are carrying. Each sensor measurement you make to a particular object has a confidence level, thus you are never totally sure whether the object you are scanning is benign or dangerous. You need to compare measurements with other robots to reduce uncertainty. You can only sense objects by moving within a radius of 1 meter around the object and activating the sensor. You can sense multiple objects each time you activate your sensor, sensing all objects within a radius of 1 meter. You can exchange text messages with other robots, although you need to be at most 5 meters away from them to receive their messages and send them messages."

        if self.openai:
            self.llm_messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
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
                        "x": {
                            "type": "integer",
                            "description": "The x coordinate of the location to move to.",
                        },
                        "y": {
                            "type": "integer",
                            "description": "The y coordinate of the location to move to.",
                        },
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "activate_sensor",
                "description": "Activates sensor. You need to be at most 1 meter away from an object to be able to sense it. Returns a list of lists, each of the sublists with the following format: [“object id”, “object x,y location”, “weight”, “benign or dangerous”, “confidence percentage”]. For example: [[“1”,”4,5”,”1”,”benign”,”0.5”],[“1”,”4,5”,”1”,”benign”,”0.5”]].",
                "parameters":{ "type": "object", "properties": {}},
                
            },
            {
                "name": "pick_up",
                "description": "Picks up an object that is 1 meter away from it at most.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "integer",
                            "description": "The ID of the object to pick up",
                        }
                    },
                    "required": ["object_id"],
                },
            },
            {
                "name": "send_message",
                "description": "Broadcasts message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text message.",
                        }
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "drop",
                "description": "Drops any object previously picked up.",
                "parameters":{ "type": "object", "properties": {}},
                
            },
            {
                "name": "scan_area",
                "description": "Returns the locations of all objects and robots in the scene.",
                "parameters":{ "type": "object", "properties": {}},
                
            },

        ]
	
        self.setup_llm(self.device)
        log_file = "log_ai.txt"
        self.log_f = open(log_file,"w")

    def control(self,messages, robotState, action_function, function_output):
        #print("Messages", messages)
        history_prompt, action_function = self.ask_llm(messages, robotState, action_function, function_output)
        #action_function = input("Next action > ").strip()
        #action_function = "scan_area()"
        action_function = "llm_control." + action_function[:-1]
    
        if not ("drop" in action_function or "activate_sensor" in action_function or "scan_area" in action_function):
            action_function += ","
            
        action_function += "robotState, next_observation, info)"


        self.action_sequence = 0
        
        return action_function

    @staticmethod
    def calculateHValue(current,dest,all_movements):

        dx = abs(current[0] - dest[0])
        dy = abs(current[1] - dest[1])
     
        
        if all_movements:   
            D = 1
            D2 = np.sqrt(2)
     
            h = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
        else:    
            h = dx + dy #For only four movements

        return h

    @staticmethod
    def tracePath(node_details,dest):
        path = []
        
        currentNode = dest

        while node_details[currentNode[0]][currentNode[1]]["parent"][0] != currentNode[0] or node_details[currentNode[0]][currentNode[1]]["parent"][1] != currentNode[1]:
            path.append(currentNode)
            currentNode = node_details[currentNode[0]][currentNode[1]]["parent"]
            
        path.reverse()
        
        return path
            
    @staticmethod
    def findPath(startNode,endNode,occMap,ignore=[],all_movements=True):

        all_movements = False

        if min(endNode) == -1 or any(endNode >= occMap.shape) or (endNode[0] == startNode[0] and endNode[1] == startNode[1]):
            return []
            
        '''
        if occMap[endNode[0],endNode[1]] != 0:
            possible_locations = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])
            found_location = False
            for p in possible_locations:
                new_location = endNode + p
                
                if min(new_location) == -1 or any(new_location >= occMap.shape):
                    continue
                
                if occMap[new_location[0],new_location[1]] == 0:
                    endNode = new_location
                    found_location = True
                    break
            
            if not found_location:
                return []
            print("changed destination to",endNode)
        '''

        openSet = [startNode]
        closedSet = []
        
        highest_cost = float('inf') #2147483647
        
        node_details = {}
        
        for s0 in range(occMap.shape[0]):
            node_details[s0] = {}
            for s1 in range(occMap.shape[1]):
                if s0 == startNode[0] and s1 == startNode[1]:
                    node_details[s0][s1] = {"f":0, "g":0, "h":0, "parent":[startNode[0],startNode[1]]}
                else:
                    node_details[s0][s1] = {"f":highest_cost, "g":highest_cost, "h":highest_cost, "parent":[-1,-1]}
        

        
        for ig in ignore: #Remove ignore nodes
            closedSet.append(tuple(ig))
        
        if all_movements:
            next_nodes = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]]) #np.array([[-1,0],[1,0],[0,1],[0,-1]]) #np.array([[1,1],[-1,1],[1,-1],[-1,-1],[-1,0],[1,0],[0,1],[0,-1]])
        else:
            next_nodes = np.array([[-1,0],[1,0],[0,1],[0,-1]])

        while openSet:
        
            currentNode = openSet.pop(0)
            closedSet.append(tuple(currentNode))
            
     
                
            for nx in next_nodes:
                neighborNode = currentNode + nx
                
                if neighborNode[0] == endNode[0] and neighborNode[1] == endNode[1]:
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    return LLMControl.tracePath(node_details, endNode)
                
                if min(neighborNode) == -1 or any(neighborNode >= occMap.shape) or not (occMap[neighborNode[0],neighborNode[1]] == 0 or occMap[neighborNode[0],neighborNode[1]] == 3) or tuple(neighborNode) in closedSet: #modified to allow a robot to step into another robot's place
                    continue

            
                gNew = node_details[currentNode[0]][currentNode[1]]["g"] + 1
                hNew = LLMControl.calculateHValue(neighborNode,endNode,all_movements)
                fNew = gNew + hNew
                
                if node_details[neighborNode[0]][neighborNode[1]]["f"] == highest_cost or node_details[neighborNode[0]][neighborNode[1]]["f"] > fNew:
                    openSet.append(neighborNode)
                    
                    node_details[neighborNode[0]][neighborNode[1]]["f"] = fNew
                    node_details[neighborNode[0]][neighborNode[1]]["g"] = gNew
                    node_details[neighborNode[0]][neighborNode[1]]["h"] = hNew
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    

        return [] #No path
        
    @staticmethod
    def position_to_action(current_pos,dest,pickup):
        
        res = np.array(dest) - np.array(current_pos) 
        
        action = -1
        
        if int(res[0]) == 0 and res[1] > 0:
            if pickup:
                action = Action.grab_left.value
            else:
                action = Action.move_left.value
        elif int(res[0]) == 0 and res[1] < 0:
            if pickup:
                action = Action.grab_right.value
            else:
                action = Action.move_right.value
        elif res[0] > 0 and int(res[1]) == 0:
            if pickup:
                action = Action.grab_up.value
            else:
                action = Action.move_up.value
        elif res[0] < 0 and int(res[1]) == 0:
            if pickup:
                action = Action.grab_down.value
            else:
                action = Action.move_down.value
        elif res[0] > 0 and res[1] > 0:
            if pickup:
                action = Action.grab_up_left.value
            else:
                action = Action.move_up_left.value
        elif res[0] < 0 and res[1] > 0:
            if pickup:
                action = Action.grab_down_left.value
            else:
                action = Action.move_down_left.value
        elif res[0] < 0 and res[1] < 0:
            if pickup:
                action = Action.grab_down_right.value
            else:
                action = Action.move_down_right.value
        elif res[0] > 0 and res[1] < 0:
            if pickup:
                action = Action.grab_up_right.value
            else:
                action = Action.move_up_right.value
        else:
            #pdb.set_trace()
            pass
            

        
        return action
        
        
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
            self.path_to_follow = LLMControl.findPath(np.array([ego_location[0][0],ego_location[1][0]]),np.array([x,y]),robotState.latest_map)
            
            if not self.path_to_follow:
                action["action"] = Action.get_occupancy_map.value
                finished = True
                output = -1
            else:
            
                next_location = [ego_location[0][0],ego_location[1][0]]
                action["action"] = LLMControl.position_to_action(next_location,self.path_to_follow[0],False)
            
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
                    action["action"] = LLMControl.position_to_action(next_location,self.path_to_follow[0],False)
                
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
            self.action_retry = 0
        
            self.action_sequence += 1    

            ob_idx = info["object_key_to_index"][str(object_id)]
         

            if not robotState.items[ob_idx]["item_weight"]:
                output = -2
                finished = True
                action["action"] = Action.get_occupancy_map.value
            else:
                location = robotState.items[ob_idx]["item_location"]
                action["action"] = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
            
        elif self.action_sequence == 1:
            if next_observation['action_status'][0] or self.action_retry == 2:
                action["action"] = Action.get_occupancy_map.value
        
                finished = True
                
                if self.action_retry == 2 and not next_observation['action_status'][0]:
                    output = -1
                else:
                    self.held_objects.append(str(object_id))
            else:
                ob_idx = info["object_key_to_index"][str(object_id)]
                location = robotState.items[ob_idx]["item_location"]
                action["action"] = LLMControl.position_to_action([ego_location[0][0],ego_location[1][0]],location,True)
                self.action_retry += 1
        


        return action,finished,output
             
    def drop(self,robotState, next_observation, info):

        action = self.sample_action_space
        action["action"] = -1
        finished = True

        output = self.held_objects
        
        action["action"] = Action.drop_object.value

        return action,finished,output
        
    def scan_area(self,robotState, next_observation, info):
        action = self.sample_action_space

        finished = True

        output = []
        
        action["action"] = Action.get_occupancy_map.value

        return action,finished,output
        
    def ask_llm(self,messages, robotState, action_function, output):


        prompt = ""
        token_output = ""
        possible_functions = ["drop","go_to_location","pick_up","send_message","activate_sensor", "scan_area"]


        if "pick_up" in action_function:
            first_index = action_function.index("(") + 1
            arguments = action_function[first_index:].split(",")
            
            if output == -1:
                prompt += "Failed to pick up object " + arguments[0] + ". "
            elif output == -2:
                prompt += "You cannot pick an unknown object. Hint: scan the area and move closer to objects. " #+ arguments[0] + ". "
            else:
                prompt += "Succesfully picked up object " + arguments[0] + ". "
                
        elif "go_to_location" in action_function:
            first_index = action_function.index("(") + 1
            arguments = action_function[first_index:].split(",")
            
            if output:
                prompt += "Failed to move to location (" + arguments[0] + "," + arguments[1] + "). "
            else:
                prompt += "Arrived at location. "
        elif "activate_sensor" in action_function:
            if output:
                prompt += "The sensing results are the following: " + str(output) + ". "
            else:
                prompt += "No objects are close enough to be sensed. Hint: scan the area and move closer to objects. "
        elif "send_message" in action_function:
            prompt += "Sent message. "
            
        elif "drop" in action_function:
            if output:
                prompt += "Dropped object " + output[0] + ". "
                self.held_objects.pop(0)
            else:
                prompt += "No object to drop. "
        elif "scan_area" in action_function:
            objects_location = np.where(robotState.latest_map == 2)
        
            """
            if objects_location[0].size > 0:
                prompt += "Last seen objects at locations: "
            """
            for ob_idx in range(len(objects_location[0])):
                
                    
                map_key = str(objects_location[0][ob_idx]) + '_' + str(objects_location[1][ob_idx])
                for m_idx,map_object in enumerate(robotState.map_metadata[map_key]):
                
                    if ob_idx or m_idx:
                        prompt +=", "
                

                    object_id = map_object[0]

                        
                    prompt += "Object " + object_id + " is at location (" + str(objects_location[0][ob_idx]) + "," + str(objects_location[1][ob_idx]) + ")"
                
            prompt += ". "
            
            robots_location = np.where(robotState.latest_map == 3)
            
            """
            if robots_location[0].size > 0:
                prompt += "Last seen robots at locations: "
            """
            for ob_idx in range(len(robots_location[0])):
            
                map_key = str(robots_location[0][ob_idx]) + '_' + str(robots_location[1][ob_idx])
                for m_idx,map_object in enumerate(robotState.map_metadata[map_key]):
                
                    if ob_idx or m_idx:
                        prompt +=", "
                

                    object_id = map_object           
              
                    prompt += "Robot " + object_id + " is at location (" + str(robots_location[0][ob_idx]) + "," + str(robots_location[1][ob_idx]) + ")"
                
            prompt += ". "
            
        action_status_prompt = prompt
        
        if self.openai and action_status_prompt:
            self.llm_messages[-1]["content"] = action_status_prompt
        
        status_prompt = ""
        
        if not robotState.object_held and self.held_objects:
            status_prompt += "Accidentally dropped object " + self.held_objects[0] + ". "
            self.held_objects.pop(0)
            
            
        if messages:

            status_prompt += "You have received the following messages: "
            
            for m in messages:
                status_prompt += "Message from " + str(m[0]) + ": '" + m[1] + "'. "
            status_prompt += ". "
            
        
        
        status_prompt += "Your current strength is " + str(robotState.strength) + ". "
        
        ego_location = np.where(robotState.latest_map == 5)
        
        status_prompt += "Your current location is " + "(" + str(ego_location[0][0]) + "," + str(ego_location[1][0]) + "). "
        
        prompt += status_prompt
        
        prompt += "What would be your next action? Write a single function call."
        
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
            while True:

                response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                        *self.llm_messages,
                        {"role": "user", "content": prompt}
                    ],
                  functions=self.llm_functions,
                  function_call="auto"
                )
                print(self.llm_messages, {"role": "user", "content": prompt})
                print(response)

                response_message = response["choices"][0]["message"]
                
                if response_message.get("function_call"):
                    self.llm_messages.append({"role": "user", "content": status_prompt})
                    
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
                    self.llm_messages.append({"role": "function", "name": function_name, "content": ""})
                    
                    for msg in self.llm_messages:
                        final_prompt += msg["role"] + " >> "
                        if "name" in msg and msg["content"]:
                            final_prompt += msg["name"] + " > "
                            
                        final_prompt += msg["content"] + '\n'
                    

                    
                    break
        
        self.log_f.write(final_prompt + function_output + '\n')
        
        return token_output+'\n', function_output
            
    def setup_llm(self,device):


        
        if self.openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            pretrained_model = "eachadea/vicuna-13b-1.1" #"tiiuae/falcon-40b-instruct" #"eachadea/vicuna-13b-1.1"
            
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16, device_map='sequential', max_memory={0: '12GiB', 1: '20GiB'}, revision='main', low_cpu_mem_usage=True, offload_folder='offload')
            
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
