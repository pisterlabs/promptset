import os
import openai
import logging
import subprocess
import re
import json
import docker
import csv
import time

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - \n%(message)s \n', level=logging.INFO)

class SolGPT:
    AI_MODEL="gpt-3.5-turbo"
    CURR_DIR = os.path.dirname(__file__)
    PARENT_DIR = os.path.dirname(CURR_DIR)
    TOKEN_AI = os.environ["TOKEN_AI"]
    openai.api_key = TOKEN_AI
    VUL_TOOLS = {
        "slither": {
            "docker_name": "slither",
            "docker_image": "trailofbits/eth-security-toolbox",
            "docker_shared_folder": os.join.path(PARENT_DIR,"cleaned" + ":/home/ethsec/shared",
            "cmd": lambda sol_file, pragma: [f"solc-select use {pragma}", f"slither shared/{sol_file} --json -"]
        },
        # Other vulnerability detection tools can be added
    }

    def __init__(self, sol_path:str, vul_tool:str="slither"):
        
        if os.path.exists(sol_path):
            self.sol_file = os.path.basename(sol_path)
            self.sol_path = os.path.dirname(sol_path)
        else:
            logging.warning(f"Provided path '{self.sol_path}' does not exist. Using the first file in the 'cleaned' folder ")
            self.sol_file = os.listdir(self.PARENT_DIR + "/cleaned")[0]
            self.sol_path = self.PARENT_DIR + "/cleaned"
        # Create a folder for the output
        self.output_dir = os.path.join(self.sol_path, f"{self.sol_file.split('.')[0]}")
        self.vul_tool = vul_tool
        self.docker_name = self.VUL_TOOLS[vul_tool]["docker_name"]
        self.docker_image = self.VUL_TOOLS[vul_tool]["docker_image"]
        self.docker_shared_folder = self.VUL_TOOLS[vul_tool]["docker_shared_folder"]
        self.cmd = self.VUL_TOOLS[self.vul_tool]["cmd"]
        self.docker_client = docker.from_env()
        self.docker_container = self.vul_tool
        self.vuls_raw = None

        def remove_comments(string):
            pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
            # first group captures quoted strings (double or single)
            # second group captures comments (//single-line or /* multi-line */)
            regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
            def _replacer(match):
                # if the 2nd group (capturing comments) is not None,
                # it means we have captured a non-quoted (real) comment string.
                if match.group(2) is not None:
                    return "" # so we will return empty to remove the comment
                else: # otherwise, we will return the 1st group
                    return match.group(1) # captured quoted-string
            return regex.sub(_replacer, string)
        
        # Remove extra tab and new line
        rem_mul_new_line = r"\n+"
        rem_mul_new_tab = r"\t+"
        logging.info(f"Reading file '{self.sol_file}'")
        self._sol_txt = list()
        with open(os.path.join(self.sol_path, self.sol_file)) as ff:
            for line in ff:
                temp = re.sub(rem_mul_new_line, "\n", remove_comments(line.strip()))
                self._sol_txt.append(re.sub(rem_mul_new_tab, "\t", temp))
        self.sol_txt = '\n'.join(self._sol_txt)
        
        # Get code only since ChatGPT may add some description not included in a comment block
        code_only = r"pragma solidity.*}"
        self.sol_txt = re.search(r"pragma solidity.*}", self.sol_txt, re.DOTALL).group(0)
        with open(os.path.join(self.sol_path, self.sol_file), "w") as ff:
            ff.write(self.sol_txt)

        # Get pragma 
        pattern = r"(?<=pragma solidity ).*;"
        regex = re.search(pattern, self.sol_txt)
        self.pragma = regex.group(0)[1:-1] if regex.group().startswith('^') else str(regex.group(0)[:-1])
        if int(self.pragma.split(".")[1]) == 4 and int(self.pragma.split(".")[2]) < 11:
            self.pragma = "0.4.11"
        logging.info(f"Pragma: {self.pragma}")

    def run_on_docker(self, cmd:list=None):
        
        if self.docker_container not in self.VUL_TOOLS.keys():
            raise ValueError(f"Container name '{self.docker_container}' not found in the list of available tools")

        client = docker.from_env()
        logging.info(f"""
        Docker container name: {self.docker_container}
        Docker image name: {self.docker_image}
        Docker shared folder: {self.docker_shared_folder}
        Docker command: {cmd}
        """)

        # Get docker container instance, else create it
        try:
            container = client.containers.get(self.docker_container)
        except docker.errors.NotFound:

            if self.docker_shared_folder is not None:
                host_path, container_path = self.docker_shared_folder.split(':')
                volumes = {host_path: {'bind': container_path, 'mode': 'rw'}}

            container = client.containers.create(
                image=self.docker_image,
                name=self.docker_shared_folder,
                volumes=volumes,
                stdin_open=True,
            )

        if container.status == "running":
            logging.debug(f"Docker container '{container.name}' is running")
        else:
            logging.info(f"Docker container '{container.name}' is in status '{container.status}'\nStarting container...")
            # while container.status != "running":
            container.start()
            logging.info(f"New state '{container.status}'")
        if cmd is None:
            cmd = self.cmd(self.sol_file, self.pragma)
        if isinstance(cmd, list):
            for c in cmd:
                _, output = container.exec_run(cmd=c, stdout=True, stderr=True, stream=False)
        else:
            raise Exception("Command must be a list of strings")

        logging.info(output)
        return output
    
    def _get_vul(self):
        vuls_raw = json.loads(self.run_on_docker())
        if self.vul_tool == "slither":
            if vuls_raw.get("results", False).get("detectors", False):
                vulns = list(map(lambda x: {key: x[key] for key in ["description", "check", "impact", "confidence","first_markdown_element"]}, 
                                 vuls_raw["results"]["detectors"]))
                logging.debug(vuls_raw)
            else:
                logging.error(f"Slither execution failed:\n\n{vuls_raw}")
                vulns = [{
                        "impact":"High",
                        "confidence":"High",
                        "description": vuls_raw["error"].replace("\n", "")
                        }]
        # Add `elif` with other vulnerability detection tools (i.e., Mythril)
        else:
            raise ValueError(f"Vulnerability tool '{self.vul_tool}' not supported")
        return vulns
    
    @classmethod
    def get_vul(cls, sol_path:str, vul_tool:str="slither"):
        inst = cls(sol_path, vul_tool)
        return inst._get_vul()

    def get_fix(
            self, 
            level:str="High", 
            with_tool=False,
            new_sol_file:str="fixed.sol",
            prompt_init:str="Fix the vulnerability in the Solidity code.\nProvide the fixed contract"
        ):
        severity = {
            "High": ["High"],
            "Medium": ["High", "Medium"],
            "Low": ["High", "Medium", "Low"]
        }
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        prompt = prompt_init
        new_code = None
        prompt += f"\"{self.sol_txt}\""
        
        # If tool is selected, the vulnerabilities' description is added to the prompt
        if with_tool:
            if self.vuls_raw is None:
                self.vuls_raw = self._get_vul()
            if self.vul_tool == "slither":
                vuls = list(filter(lambda x: x["impact"] in severity[level], self.vuls_raw))
            
            if len(vuls) > 0:
                logging.info(f"Found {len(vuls)} vulnerabilities of severity {level}")
            
            prompt += f'\nErrors:\n'
            for item in vuls:
                prompt += f'- {item["description"]}\n'
        try:
            logging.debug(f"Prompt: {prompt}")
            completions = openai.ChatCompletion.create(
                model=self.AI_MODEL,
                temperature=0.1,
                messages = [{
                                "role": "user",
                                "content": f"{prompt}"
                            }]
            )
            new_code = completions['choices'][0]['message']['content']
            logging.info(f"NEW CODE: {new_code}")

            code_only = r"pragma solidity.*}"
            new_code = re.search(r"pragma solidity.*}", new_code, re.DOTALL).group(0)
            # Save the file with the new Solidity code (hopefully) provided by ChatGPT
            with open(os.path.join(self.output_dir, new_sol_file), "w") as ff:
                ff.write(new_code)
        except Exception as e:
            new_code = e
            logging.error(f"OpenAI execution failed\n{type(e)}\n\n{e}")    
        return new_code
     
