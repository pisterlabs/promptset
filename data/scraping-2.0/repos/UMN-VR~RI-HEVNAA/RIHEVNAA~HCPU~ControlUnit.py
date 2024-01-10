import asyncio
import sys
import os
import time
import numpy as np
import logging
from multiprocessing import Queue
import json
from langchain import PromptTemplate
import RIHEVNAA.TaskGeneration.creative_task_generation as creative_task_generation


# define the states of the A-CPU
STATE_IDLE = 'idle'
STATE_RUNNING = 'running'
STATE_HALTED = 'halted'

class ControlUnit:

    def load_chain(self, chain):
        # Add a chain to the A-CPU's chain dictionary
        # The chain's ID is used as the key

        #Open the JSON file
        with open(chain, 'r') as f:
            #Load the JSON file as a dictionary
            chain = json.load(f)

        self.chains.append(chain)


    def load_chains(self, folder):
        #A chain is saved as a JSON file  
        #Load all JSON files in the folder
        chains = []
        
        # Get all subfolders in the folder
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        # for each subfolder in the folder
        for subfolder in subfolders:
            # Get all files in the subfolder
            files = [f.path for f in os.scandir(subfolder) if f.is_file()]
            # for each file in the subfolder
            for file in files:
                # if the file is a JSON file
                if file.endswith(".json"):

                    #if file is 'chain.json' 
                    if file == "chain.json":
                        # load the chain
                        self.load_chain(file)
                        print(f"Loaded chain: {file}")
                    
                    else: 
                        # load the sub-chain
                        self.load_chain(file)
                        print(f"Loaded sub-chain: {file}")

    
        

    def __init__(self, name, bus_manager):
        self.name = name+"-ControlUnit"
        self.bus_manager = bus_manager
        self.state = STATE_IDLE
        self.task_queue = Queue()
        self.chains = []
        self.running_task = None
        self.running_chain = None

        self.load_chains("chains")
    

    #async def run(self, logger):
    async def run(self, logger, RAM, Accelerators):
        logger.info(f"@{self.name}.run")

        #Init return values
        instruction = None
        program = None
        memory = None

        if self.state == STATE_IDLE:
            # If HCPU is idle, check the task queue for new tasks
            if self.task_queue.empty():
                # If there are no new tasks, start creative task generation
                logger.info(f"Task Queue is empty. Starting creative task generation")

                # Generate a new task
                self.running_task = await creative_task_generation.create_task(logger, RAM, Accelerators)
                logger.info(f"New task generated: {self.running_task}")


                if self.running_task != None:
                    # Set the HCPU state to running
                    self.state = STATE_RUNNING
                    logger.info(f"HCPU state set to {self.state}")

                else:
                    logger.info(f"Creative Task Generation Failed.")
                    # Set the HCPU state to idle
                    self.state = STATE_IDLE
                    logger.info(f"HCPU state set to {self.state}")

            else:
                # If there are new tasks, start executing the tasks
                logger.info(f"New task found in Task Queue. Starting execution")
                # Set the HCPU state to running
                self.state = STATE_RUNNING
                self.running_task = self.task_queue.get()
            
        
        if self.state == STATE_RUNNING:

            # if running_chain is DONE
            if self.running_chain.isDONE:

                # if running_task is DONE
                if self.running_task.isDONE:
                    # Set the HCPU state to idle
                    self.state = STATE_IDLE
                    # Set running_task to None
                    self.running_task = None

                    if self.running_chain.next_chain != None:
                        # Set running_chain to the next chain in the task
                        self.running_chain = self.running_chain.next_chain
                else: # else if running_task is not DONE
                    # run running_task
                    instruction = self.running_task.instruction #"EXECUTE.py"
                    program = self.running_task.program
                    memory = self.running_task.memory



        
        
        logger.info(f"@{self.name}.run: Done")
        return instruction, program, memory