__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."


from typing import Any, Dict, List, Optional, Union, AnyStr
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.schema import LLMResult
from pathlib import Path
from datetime import datetime
import re
import logging

"""
Class to illustrate the various callbacks
"""


class LLMFileCallback(BaseCallbackHandler):
    def __init__(self,
                 path: Path,
                 print_prompts: bool = False,
                 print_class: bool = False,
                 title: Optional[str] = "log",
                 color: Optional[str] = "blue"
                 ) -> None:
        self.color = color
        self.print_prompts = print_prompts
        self.print_class = print_class
        self.path = path
        self.file_handle = open(path, 'w')
        self.title = title
        self.texts = []
        self.output_keys = []
        self.output_values = []

    def on_llm_start(self, serialized: Dict[AnyStr, Any], prompts: List[AnyStr], **kwargs: Any) -> None:
        if self.print_prompts:
            self.file_handle.write("Prompts ----------------\n")
            for prompt in prompts:
                self.file_handle.write(f"{prompt}\n")
            self.file_handle.write("\n")
            self.file_handle.flush()
            self.file_handle.write(f"-------------\n\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.file_handle.write("End LLM")
        pass

    def on_llm_new_token(self, token: AnyStr, **kwargs: Any) -> None:
        logging.warning(f'Call back on token {token} should not be called')
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.warning(f'Call back on llm_langchain error should not be called')
        pass

    def on_chain_start(self, serialized: Dict[AnyStr, Any], inputs: Dict[AnyStr, Any], **kwargs: Any) -> None:
        if self.print_class:
            self.file_handle.write(f"Class -------------------\n")
            class_name = serialized["name"]
            self.file_handle.write(class_name)
            self.file_handle.write("\n-----------------\n\n")
            self.file_handle.flush()

    def on_chain_end(self, outputs: Dict[AnyStr, Any], **kwargs: Any) -> None:
        self.file_handle.write("Output --------------------------------------\n")
        keys = []
        values = []
        for k, v in outputs.items():
            keys.append(k)
            values.append(v)
            self.file_handle.write(f"{k}:\n")
            self.file_handle.write(f"{v}\n\n")
        self.output_keys.append(keys)
        self.output_values.append(values)
        self.file_handle.write("------------------------------\n")
        self.file_handle.flush()

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.warning(f'Call back on error onm chain should not be called')
        pass

    def on_tool_start(self, serialized: Dict[AnyStr, Any], input_str: AnyStr,**kwargs: Any) -> None:
        self.file_handle.write(datetime.today().strftime('%Y-%m-%d'))
        self.file_handle.write("\n========")
        self.file_handle.flush()


    def on_agent_action(self, action: AgentAction, color: Optional[AnyStr] = None, **kwargs: Any) -> Any:
        self.file_handle.write(f">>> action: {action.log}")

    def on_tool_end(
            self,
            output: AnyStr,
            color: Optional[AnyStr] = None,
            observation_prefix: Optional[AnyStr] = None,
            llm_prefix: Optional[AnyStr] = None,
            **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            self.file_handle.write(f"\n{observation_prefix}")
        self.file_handle.write(output)
        if llm_prefix is not None:
            self.file_handle.write(f"\n{llm_prefix}")
        self.file_handle.flush()

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        logging.warning('Call back on error on tool should not be called')
        pass

    def on_text(self ,text: AnyStr, color: Optional[AnyStr] = None, end: AnyStr = "", **kwargs: Any ) -> None:
        self.file_handle.write("Text ----------------------------\n")
        self.file_handle.write(text)
        self.file_handle.flush()
        self.file_handle.write("--------------------------\n\n")
        self.texts.append(text)

    def on_agent_finish(self, finish: AgentFinish, color: Optional[AnyStr] = None, **kwargs: Any) -> None:
        self.file_handle.write(f"{finish.log}\n")
        self.file_handle.flush()
        self.file_handle.close()

    def create_html(self) -> None:
        table: str = """
<table class="table table-striped">
    <tr>
        <th>
            Agent
        </th>
        <th>
            Input
        </th>
        <th>
            Output
        </th>
    </tr>
        """
        for text, keys, values in zip(self.texts, self.output_keys, self.output_values):
            table += f"""
    <tr>
        <td>
            {extract_agent(text)}
        </td>
        <td>
            {extract_input(text)}
        </td>
        <td>
            <pre>{"<br />".join(values)}</pre>
        </td>
    </tr>
            """
        table += "</table>"

        target_file = f"{self.path.stem}.html"
        with open(target_file, "w") as f:
            f.write(f"""
<html>
    <head>
        <meta charset="UTF-8" />
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>
        <style>
            pre {{
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1>{self.title}</h1>
            <h2>{generate_timestamp()}</h2>
            {table}
        </div>
    </body>
</html>
            """)
        print(f"Saved chat content to {target_file}")


def generate_timestamp() -> AnyStr:
    # Get the current date and time
    now = datetime.now()

    # Get the weekday, day, month, year, and time in English
    weekday = now.strftime("%A")
    day = now.strftime("%d")
    month = now.strftime("%B")
    year = now.strftime("%Y")
    time = now.strftime("%H:%M:%S")

    # Create the timestamp string
    timestamp = f"{weekday}, {day} {month} {year} {time}"

    return timestamp

def extract_input(text: AnyStr) -> AnyStr:
    return re.sub(r".+?'input':\s*'(.+)'}", r"\1", text)

def extract_agent(text: AnyStr) -> AnyStr:
    return re.sub(r"(.+?)\:.+", r"\1", text)