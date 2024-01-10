import os
import re
import subprocess
from io import StringIO

import openai
import pylint.lint
import yaml
from pylint.reporters.text import TextReporter
from util import logger

import pygpt

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)


class StaticAnalysisRunner:
    def __init__(self):
        # Initialize the class
        self.analyzers = [Flake8StaticAnalysis(), OpenAIStaticAnalysis()]
        # self.analyzers = [OpenAIStaticAnalysis()]

    def analyze(self, code):
        # Perform static analysis on the code
        # Return the analysis results
        results = []

        for analyzer in self.analyzers:
            results.append(analyzer.analyze(code))

        return results


class StaticAnalysis:
    def __init__(self):
        # Initialize the class
        return

    def analyze(self, code):
        # Perform static analysis on the code
        return


class PyLintStaticAnalysis(StaticAnalysis):
    def __init__(self):
        super().__init__()

    def analyze(self, code):
        # Initialize the PyLint API
        output = StringIO()
        reporter = TextReporter(output)
        options = [
            "--disable=all",
            "--enable=warning,refactor,convention",
            "--output-format=text",
        ]

        # Run the PyLint analysis on the code
        # pylint_opts = pylint.config.Options.from_command_line(options)

        # pylint.lint.Run(code, reporter=reporter, exit=False, options=pylint_opts)

        pylint.lint.Run(
            [
                "--disable=all",
                "--enable=convention",
                "--enable=refactoring",
                "--enable=warning",
                "--output-format=text",
                "-",
            ],
            from_input_string=code,
            do_exit=False,
        )

        # Parse the analysis results and return them as a string
        results = output.getvalue()
        output.close()
        return results


class OpenAIStaticAnalysis(StaticAnalysis):
    def __init__(self):
        super().__init__()
        # Initialize OpenAI API credentials and parameters
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.prompt = "Please analyze the following code:\n"

    def analyze(self, code):
        # Prepare the prompt and input for the OpenAI API
        input_text = self.prompt + code.strip()

        # Call the OpenAI API to generate the analysis
        response = pygpt.create(
            engine=config["model_engine1"],
            prompt=input_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract the analysis results from the API response
        analysis_text = response.choices[0].text
        analysis_text = re.sub(r"\n+", "\n", analysis_text.strip())

        return analysis_text


class Flake8StaticAnalysis(StaticAnalysis):
    def __init__(self):
        super().__init__()
        # Initialize the Flake8 command
        self.cmd = ["flake8", "-"]

    def analyze(self, code):
        # Call Flake8 to analyze the code
        p = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate(input=code.encode())
        result = stdout.decode().strip()
        if result == "":
            return "No issues found"
        else:
            return result


class ProspectorStaticAnalysis(StaticAnalysis):
    def __init__(self):
        super().__init__()
        # Initialize the Prospector command
        self.cmd = ["prospector", "-"]

    def analyze(self, code):
        # Call Prospector to analyze the code
        p = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate(input=code.encode())
        result = stdout.decode().strip()
        if result == "":
            return "No issues found"
        else:
            return result
