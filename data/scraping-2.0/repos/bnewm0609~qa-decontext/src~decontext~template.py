from pathlib import Path
from typing import Union, List

from jinja2 import StrictUndefined
from jinja2 import Template as JTemplate
import yaml

from decontext.data_types import OpenAIChatMessage


class Template:
    def __init__(self, template: Union[Path, str, List]):
        self.template = self.load_template(template)

        if isinstance(self.template, str):
            self.template_type = "completion"
        else:
            self.template_type = "openai_chat"

    def load_template(
        self, template: Union[Path, str, List]
    ) -> Union[List[OpenAIChatMessage], str]:
        """Load the template from the config.

        The passed template takes one of three forms:
        1. a list[dict[str, str]] (for the OpenAI Chat Endpoint). The keys are the role ("user", "system")
            and the values are the template for that message. The dict[str, str] is converted into an
            `OpenAIChatMessage`.
        2. a string containing the template (for the OpenAI Completion or Claude endpoints)
        3. a string with a yaml filepath to either of the two above template types.
        The template strings are jinja templates.

        Args:
            template (Union[str, list]): The template or a path to a yaml file with template.

        Returns:
            list[OpenAIChatMessage] for the OpenAI Chat API case and a str for the Completion or Claude
            cases with the template. The template is not filled at this point.
        """

        loaded_template: Union[List[OpenAIChatMessage], str] = ""
        # there are a few choices for template:
        if isinstance(template, Path) or (
            isinstance(template, str)
            and len(template) < 256
            and Path(template).is_file()
        ):
            template = str(template)
            # the template is a path to a file - read the template from the file
            with open(template) as f:
                if template.endswith("yaml"):
                    loaded_template = yaml.safe_load(f)["template"]
                    if isinstance(
                        loaded_template, list
                    ):  # we're using OpenAI chat
                        loaded_template = [
                            OpenAIChatMessage(**item)
                            for item in loaded_template
                        ]
                else:
                    loaded_template = f.read()
        elif isinstance(template, str):
            # assume the template is for a non-chat model. It also could be a path that's misspelled.
            if template.endswith(".yaml") or template.endswith(".txt"):
                raise FileNotFoundError(
                    f"Make sure path is correct. Unable to find this file: {template}"
                )
            loaded_template = template
        elif isinstance(
            template, list
        ):  # or isinstance(template, ListConfig):
            # assume that the passsed thing is the template dict itself
            loaded_template = [OpenAIChatMessage(**item) for item in template]
        else:
            raise ValueError(
                "Template must be either a string, list or path to a valid file"
            )

        return loaded_template

    def fill(self, fields: dict) -> Union[List[OpenAIChatMessage], str]:
        result: Union[List[OpenAIChatMessage], str] = ""
        if self.template_type == "completion":
            result = JTemplate(
                self.template, undefined=StrictUndefined
            ).render(fields)
        elif isinstance(self.template, list):
            result = []
            for chat_message in self.template:
                new_message_content = JTemplate(
                    chat_message.content, undefined=StrictUndefined
                ).render(
                    fields
                )  # any extra elements will be ignored
                result.append(
                    OpenAIChatMessage(
                        role=chat_message.role,
                        content=new_message_content,
                    )
                )
        return result
