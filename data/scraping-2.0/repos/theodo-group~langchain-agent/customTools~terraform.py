import os
import subprocess
from langchain.tools.base import BaseTool

class TerraformValidateTool(BaseTool):
    name = "terraform-validate"
    description = "A tool for validating that terraform code is valid. MUST be ran for each terraform code generated. Takes in a string of terraform code"

    async def _arun(self, code: str) -> str:
        return self.run(code)

    def _run(self, code: str) -> str:
        with open("temp_terraform_file.tf", "w") as file:
            file.write(code)
        
        try:
            output = subprocess.check_output(["terraform", "validate"], stderr=subprocess.STDOUT, text=True)
            return f"Validation successful: {output}"
        except subprocess.CalledProcessError as e:
            return f"Validation failed: {e.output}"
        finally:
            os.remove("temp_terraform_file.tf")