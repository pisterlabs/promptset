import itertools
import pathlib
import time
import typing

import openai
from bardapi import Bard
from loguru import logger
from pydantic import BaseModel

from git_file_keyword.config import ExtractConfig
from git_file_keyword.plugin import BasePlugin
from git_file_keyword.result import Result, FileResult
from git_file_keyword.utils import split_list


class Ask(BaseModel):
    file_list: typing.List[pathlib.Path]
    request_txt: str


class BaseLLMPlugin(BasePlugin):
    # todo: LLM usage takes some money. Maybe the results should be cached.
    prompt = """
Generate concise (<30 words) descriptions for each source file based on their associated keywords, 
summarizing/guessing the function of each file. 
Avoid comments, quotes, or explanations.

Sample Input:

```
- somedir/clipboard.java: cut, paste, auto-sync
- somedir/webview.java: ...
```

Sample Output:

```
- somedir/clipboard.java: A auto-sync clipboard implementation contains cut/paste.
- somedir/webview.java: ...
```
        """

    def plugin_id(self) -> str:
        return "llm"

    def gen_ask_group(self, result: Result) -> typing.Dict[pathlib.Path, Ask]:
        merged_dict: typing.Dict[pathlib.Path, typing.Dict[pathlib.Path, FileResult]] = {}
        for file_path, file_result in result.file_results.items():
            dir_path = file_path.parent
            if dir_path not in merged_dict:
                merged_dict[dir_path] = dict()
            merged_dict[dir_path][file_path] = file_result

        ask_dict = dict()
        for directory_path, group_dict in merged_dict.items():
            # if the whole group does not contain any changes, ignore it
            if all((each.cached for each in group_dict.values())):
                continue

            logger.info(f"prepare llm response for {directory_path}")
            # send the group to AI
            if len(group_dict) < 10:
                lines = [
                    f"- {filepath}: {list(result.keywords)}"
                    for filepath, result in group_dict.items()
                ]
                request_txt = "\n".join(lines)
                ask_dict[directory_path] = Ask(
                    file_list=list(group_dict.keys()), request_txt=request_txt
                )
            else:
                # split into multi parts
                chunks = list(split_list(list(group_dict.items()), 10))
                for i, chunk in enumerate(chunks, start=1):
                    lines = [
                        f"- {filepath}: {list(result.keywords)}"
                        for filepath, result in chunk
                    ]
                    request_txt = "\n".join(lines)

                    ask_dict[pathlib.Path(f"{directory_path.as_posix()}_{i}")] = Ask(
                        file_list=[filepath for filepath, _ in chunk],
                        request_txt=request_txt,
                    )
        return ask_dict


class BardLLMPlugin(BaseLLMPlugin):
    # todo: current bard model looks too bad to use ...
    #  and there is no official sdk
    token = ""
    proxies = {
        "http": "http://127.0.0.1:7890",
    }

    def __init__(self):
        self.bard = Bard(
            token=self.token,
            proxies=self.proxies,
        )

    def plugin_id(self) -> str:
        return "llm-bard"

    def apply(self, config: ExtractConfig, result: Result):
        ask_dict = self.gen_ask_group(result)
        answer_dict = dict()

        for each_dir, each_ask in ask_dict.items():
            resp = self.bard.get_answer(f"{self.prompt}\n{each_ask.request_txt}")
            responses = resp["content"]

            valid_lines = [
                each for each in responses.split("\n") if each.startswith("-")
            ]
            for line in valid_lines:
                file_path, description = line.split(": ", 1)
                file_path = file_path.lstrip("- ")

                # fix this file_path
                # model sometimes returns the name only
                for each_valid_file_path in each_ask["file_list"]:
                    if file_path == each_valid_file_path.name:
                        file_path = each_valid_file_path
                        break
                else:
                    # maybe it's correct ...
                    file_path = pathlib.Path(file_path)

                answer_dict[file_path] = description

        # update to result
        for each_path, each_desc in answer_dict.items():
            if each_path not in result.file_results:
                logger.warning(f"{each_path} not in result")
                continue

            result.file_results[each_path].plugin_output[self.plugin_id()] = each_desc


class OpenAILLMPlugin(BaseLLMPlugin):
    token = ""
    model = "gpt-3.5-turbo"
    rate_limit_wait = 20

    def plugin_id(self) -> str:
        return "llm-openai"

    def apply(self, config: ExtractConfig, result: Result):
        ask_dict = self.gen_ask_group(result)
        answer_dict = dict()
        openai.api_key = self.token

        logger.info(f"total llm requests to go: {len(ask_dict)}")
        for cur, (each_dir, each_ask) in enumerate(ask_dict.items()):
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt,
                    },
                    {"role": "user", "content": each_ask.request_txt},
                ],
            )
            responses: str = completion.choices[0].message.content

            valid_lines = [
                each for each in responses.split("\n") if each.startswith("-")
            ]
            for line in valid_lines:
                file_path, description = line.split(": ", 1)
                file_path = file_path.lstrip("- ")

                # fix this file_path
                # model sometimes returns the name only
                for each_valid_file_path in each_ask.file_list:
                    if file_path == each_valid_file_path.name:
                        file_path = each_valid_file_path
                        break
                else:
                    # maybe it's correct ...
                    file_path = pathlib.Path(file_path)

                answer_dict[file_path] = description

            # by default, trial api key rate limit: 3/min
            # means 20s / request
            logger.info(f"{cur + 1}/{len(ask_dict)} finished, resp: {responses}")
            time.sleep(self.rate_limit_wait)

        # update to result
        for each_path, each_desc in answer_dict.items():
            if each_path not in result.file_results:
                logger.warning(f"{each_path} not in result")
                continue

            result.file_results[each_path].description = each_desc
            result.file_results[each_path].plugin_output[self.plugin_id()] = each_desc
