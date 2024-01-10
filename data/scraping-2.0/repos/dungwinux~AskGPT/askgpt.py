import openai
import logging
from typing import List, Tuple

from volatility3.framework import exceptions, interfaces, renderers
from volatility3.framework.configuration import requirements
from volatility3.framework.objects import utility
from volatility3.plugins.windows import cmdline, pslist


vollog = logging.getLogger(__name__)


class AskGPT(interfaces.plugins.PluginInterface):
    """Ask ChatGPT about the potential user of the machine based on the image."""

    _required_framework_version = (2, 0, 0)
    _version = (0, 0, 1)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        # Since we're calling the plugin, make sure we have the plugin's requirements
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Windows kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.VersionRequirement(
                name="cmdline", component=cmdline.CmdLine, version=(1, 0, 0)
            ),
            requirements.VersionRequirement(
                name="pslist", component=pslist.PsList, version=(2, 0, 0)
            ),
            requirements.StringRequirement(
                name="model_id",
                description="OpenAI ChatGPT model select",
                optional=True,
                default="gpt-3.5-turbo",
            ),
        ]

    def _generator(self, procs):
        kernel = self.context.modules[self.config["kernel"]]
        filter_proc = [
            r":\WINDOWS\system32\svchost.exe",
            r"\SystemRoot\System32\smss.exe",
            r"%SystemRoot%\system32\csrss.exe",
            r":\WINDOWS\system32\services.exe",
            r":\Windows\System32\WUDFHost.exe",
            r"dwm.exe",
            r":\Windows\System32\RuntimeBroker.exe",
            r":\WINDOWS\system32\conhost.exe",
            r":\WINDOWS\system32",
        ]

        for proc in procs:
            result_text: str = "Unknown"
            proc_id = proc.UniqueProcessId
            process_name = utility.array_to_string(proc.ImageFileName)
            try:
                result_text = cmdline.CmdLine.get_cmdline(
                    self.context, kernel.symbol_table_name, proc
                )
            except exceptions.SwappedInvalidAddressException as exp:
                result_text = f"Required memory at {exp.invalid_address:#x} is inaccessible (swapped)"
                continue
            except exceptions.PagedInvalidAddressException as exp:
                result_text = f"Required memory at {exp.invalid_address:#x} is not valid (process exited?)"
                continue
            except exceptions.InvalidAddressException as exp:
                result_text = "Process {}: Required memory at {:#x} is not valid (incomplete layer {}?)".format(
                    proc_id, exp.invalid_address, exp.layer_name
                )
                continue

            checking = [result_text.upper().find(f.upper()) for f in filter_proc]
            checking.sort()
            if checking[-1] != -1:
                continue

            yield (0, (process_name, result_text))

    def ask(self, procs) -> Tuple[str, str]:
        """Send information to ChatGPT and ask for answer"""

        table = ""
        unique_proc = set()
        for _, (process_name, cmdline) in procs:
            cmdline: str
            if process_name in unique_proc:
                continue
            unique_proc.add(process_name)
            if cmdline.startswith('"'):
                cmdline = cmdline[1 : cmdline.index('"', 1)]
            else:
                cmdline = cmdline.split(" ")[0]
            table += process_name + "\t" + cmdline + "\n"

        # print(table)

        user_question = "Given process list above, do you know what the computer is being used for? And does it contain any known malware?"
        cur_content = table + "\n" + user_question

        model_id = self.config["model_id"]
        completion = openai.ChatCompletion.create(
            model=model_id, messages=[{"role": "user", "content": cur_content}]
        )
        response = completion.choices[0].message.content

        # Return string result from ChatGPT
        return (table, response)

    def run(self):
        kernel = self.context.modules[self.config["kernel"]]
        procs = self._generator(
            pslist.PsList.list_processes(
                context=self.context,
                layer_name=kernel.layer_name,
                symbol_table=kernel.symbol_table_name,
            )
        )

        return renderers.TreeGrid(
            [("Input", str), ("Answer", str)],
            [(0, self.ask(procs))],
        )
