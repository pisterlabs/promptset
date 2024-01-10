import io
import os

import openai
import pysnooper
import yaml
from util import logger

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)

root_folder = config["root_folder"]


class DebugRunner:
    def __init__(self):
        # Initialize the class
        self.debuggers = [PySnooperDebug()]

    def run_debug(self, code, analysis_results, test_results):
        # Perform repairs on the code based on the analysis and test results
        # Return the repaired code
        results = []
        for debugger in self.debuggers:
            result = debugger.run_debug(code, analysis_results, test_results)
            results.append(result)
        return results


class Debug:
    def __init__(self):
        # Initialize the class
        return

    def run_debug(self, code, analysis_results, test_results):
        # Perform repairs on the code based on the analysis and test results
        # Return the repaired code
        return


class PySnooperDebug(Debug):
    def annotate_all_functions(self, code):
        try:
            # Execute the code in a temporary namespace
            namespace = {}
            exec(code, namespace)

            # Find all the functions in the namespace and annotate them
            for name, value in namespace.items():
                if callable(value) and not isinstance(value, pysnooper.Tracer):
                    namespace[name] = pysnooper.snoop()(value)

            # Return the annotated code as a string
            return compile(
                "\n".join(line for line in code.splitlines()),
                "<string>",
                "exec",
            )
        except Exception as e:
            ## return the exec exception error message
            return "An error occurred: " + str(e)

    @pysnooper.snoop()
    def run_debug(self, code, analysis_results, test_results):
        # Use pysnooper to debug the code
        # Return the debug information
        try:
            annotated_code = self.annotate_all_functions(code)
            # Define a string buffer to capture the output
            output_buffer = io.StringIO()

            # Use the buffer as the stdout for pysnooper
            with pysnooper.snoop(output=output_buffer):
                # Execute the code string
                exec(annotated_code)

            return output_buffer.getvalue()
        except Exception as e:
            ## return the exec exception error message
            return "An error occurred: " + str(e)
