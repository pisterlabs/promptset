from typing import Optional

from docker.models.containers import Container
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool

DEFAULT_WORKING_DIRECTORY = "/app"
SUCCESS_MESSAGE = "The command successfully executed."
ERROR_MESSAGE = "The command failed to execute."
MESSAGE = """{message} {result}"""


class DockerShellTool(BaseTool):
    name = "shell"
    description = (
        "Useful when you need to execute linux shell commands "
        "in a specific working directory. "
    )
    container: Container = None

    def _run(
        self,
        command: str,
        working_directory_absolute_path: str = DEFAULT_WORKING_DIRECTORY,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # That approach is better because allows to interract with the process
        # process = subprocess.Popen(
        #     f"docker exec -it {self.container_id} ls {path}".split(" "),
        #     shell=False,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE
        # )
        # res = process.stdout.readlines().decode("utf-8").split("\n")

        exit_code, output = self.container.exec_run(
            f"bash -c '{command}'",
            workdir=working_directory_absolute_path,
            # If ``stream=True``, a generator yielding response chunks.
        )
        output = output.decode("utf-8").strip()
        return MESSAGE.format(
            message=(
                SUCCESS_MESSAGE
                if exit_code == 0
                else ERROR_MESSAGE
            ),
            result=(
                ""
                if output == ""
                else f"Result: {output}"
            ),
        ).strip()
