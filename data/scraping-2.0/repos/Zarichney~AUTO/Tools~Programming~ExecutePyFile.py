# /Tools/ExecutePyFile.py

import os
import subprocess
import sys
from instructor import OpenAISchema
from pydantic import Field
from Utilities.Config import WORKING_DIRECTORY
from Utilities.Log import Debug, Log, type
import pkg_resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agency.Agency import Agency

class ExecutePyFile(OpenAISchema):
    """
    Run local python (.py) file.
    Execution in this environment is safe and has access to all standard Python packages and the internet.
    Only use this tool if you understand how to troubleshoot with python, if not seek delegation to a more specialized agent.
    Additional packages can be installed by specifying them in the required_packages field.
    """

    file_name: str = Field(
        ..., 
        description="The path to the .py file to be executed."
    )
    directory: str = Field(
        default=WORKING_DIRECTORY,
        description="The path to the directory where to file is stored. Path can be absolute or relative."
    )
    parameters: str = Field(
        default="",                           
        description="Comma separated list of parameters to be passed to the script call."
    )
    required_packages: str = Field(
        default="",                           
        description="Required packages to be installed. List of comma delimited strings. Will execute ''pip install <package>'' for each package supplied"
    )

    def check_dependencies(self, python_path, required_packages):
        """Check if the required modules are installed."""

        packages = required_packages.split(',')
        
        for package in packages:
            try:
                dist = pkg_resources.get_distribution(package)
                Log(type.ACTION,"{} ({}) is installed".format(dist.key, dist.version))
            except pkg_resources.DistributionNotFound:
                Log(type.ACTION,f"The {package} module is not installed. Attempting to install...")
                try:
                    subprocess.check_call([python_path, "-m", "pip", "install", package])
                    Log(type.ACTION,f"Successfully installed {package}.")
                    
                except subprocess.CalledProcessError as e:
                    message = f"Failed to install {package}. Error: {e.output}"
                    Log(type.ERROR, message)
                    return message

        return "All required modules are installed."

    def run(self, agency: 'Agency'):
        """Executes a Python script at the given file path and captures its output and errors."""
            
        # Get the path of the current Python interpreter
        python_path = sys.executable
    
        # Check if the required modules are installed
        if self.required_packages:
            Debug(f"Agent called self.required_packages: {self.required_packages}")
            check_result = self.check_dependencies(python_path, self.required_packages)
            if check_result != "All required modules are installed.":
                return check_result
        
        # If file doesnt exist, return message
        if not os.path.exists(self.directory + self.file_name):
            Log(type.ERROR, f"Cannot execute file, incorrect path invoked: {self.directory + self.file_name}")
            return f"No file found at '{self.directory + self.file_name}'. Perhaps specify the correct path?"
        
        Log(type.ACTION, f"Executing {self.file_name}...")
        Debug(f"Agent called subprocess.run with:\n{[python_path, self.directory + self.file_name] + self.parameters.split(',')}")
        
        try:
            execution = subprocess.run(
                [python_path, self.directory + self.file_name] + self.parameters.split(','),
                text=True,
                capture_output=True,
                check=True,
                timeout=10
            )
            
            Debug(f"Agent execution cwd: {execution.cwd}")
            Debug(f"Agent execution args: {execution.args}")
            Debug(f"Agent execution results: {execution.stdout}")
            Debug(f"Agent execution errors: {execution.stderr}")
            Debug(f"Agent execution return code: {execution.returncode}")
            
            result = f"Execution results: {execution.stdout}"
            Log(type.RESULT, result)
            return result

        except subprocess.TimeoutExpired:
            result = "Execution timed out. The script may have been waiting with a prompt."
            Log(type.ERROR, result)
            return result

        except subprocess.CalledProcessError as e:
            result = f"Execution error occurred: {e.stderr}.\nPlease attempt to rectify"
            Log(type.ERROR, result)
            return result
            