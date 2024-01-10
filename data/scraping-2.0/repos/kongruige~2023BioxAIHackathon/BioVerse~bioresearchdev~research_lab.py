import os
import re
import shutil
import signal
import subprocess
import time
from typing import Dict

import openai
import requests
from bioresearchdev.roster import Roster
from bioresearchdev.utils import log_and_print_online

class ResearchEnvConfig:
    def __init__(self, clear_structure,
                 gui_design,
                 git_management,
                 incremental_develop):
        self.clear_structure = clear_structure
        self.gui_design = gui_design
        self.git_management = git_management
        self.incremental_develop = incremental_develop

    def __str__(self):
        string = ""
        string += "ResearchEnvConfig.clear_structure: {}\n".format(self.clear_structure)
        string += "ResearchEnvConfig.git_management: {}\n".format(self.git_management)
        string += "ResearchEnvConfig.gui_design: {}\n".format(self.gui_design)
        string += "ResearchEnvConfig.incremental_develop: {}\n".format(self.incremental_develop)
        return string
    
    
class ResearchEnv:
    def __init__(self, research_env_config: ResearchEnvConfig):
        self.config = research_env_config
        self.roster = Roster()
        
        # store environment information in dictionary
        self.env_dict = {
            "directory": "",
            "task_prompt": "",
            "modality": "",
            "ideas": "",
            "language": "",
            "review_comments": "",
            "error_summary": "",
            "test_reports": ""
        }
            
    
    @staticmethod
    def fix_module_not_found_error(test_reports):
        """
        Handle installing requirements when module not found
        Args:
            test_reports (_type_): _description_
        """
        if "ModuleNotFoundError" in test_reports:
            for match in re.finditer(r"No module named '(\S+)'", test_reports, re.DOTALL):
                module = match.group(1)
                subprocess.Popen("pip install {}".format(module), shell=True).wait()
                log_and_print_online("**[CMD Execute]**\n\n[CMD] pip install {}".format(module))
                
    
    def recruit(self, agent_name: str):
        self.roster._recruit(agent_name)
        
        
    def exist_employee(self, agent_name: str) -> bool:
        return self.roster._exist_employee(agent_name)

    def print_employees(self):
        self.roster._print_employees()
        
    def write_meta(self) -> None:
        directory = self.env_dict['directory']

        if not os.path.exists(directory):
            os.mkdir(directory)
            print("{} Created.".format(directory))

        meta_filename = "meta.txt"
        with open(os.path.join(directory, meta_filename), "w", encoding="utf-8") as writer:
            writer.write("{}:\n{}\n\n".format("Task", self.env_dict['task_prompt']))
            writer.write("{}:\n{}\n\n".format("Config", self.config.__str__()))
            writer.write("{}:\n{}\n\n".format("Roster", ", ".join(self.roster.agents)))
            
            # TODO: do proper writing for environment metadata
            # writer.write("{}:\n{}\n\n".format("Modality", self.env_dict['modality']))
            # writer.write("{}:\n{}\n\n".format("Ideas", self.env_dict['ideas']))
            # writer.write("{}:\n{}\n\n".format("Language", self.env_dict['language']))
            # writer.write("{}:\n{}\n\n".format("Code_Version", self.codes.version))
            # writer.write("{}:\n{}\n\n".format("Proposed_images", len(self.proposed_images.keys())))
            # writer.write("{}:\n{}\n\n".format("Incorporated_images", len(self.incorporated_images.keys())))
        print(os.path.join(directory, meta_filename), "Wrote")