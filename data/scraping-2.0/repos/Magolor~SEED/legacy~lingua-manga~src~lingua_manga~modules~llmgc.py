from .default import *
from langchain.prompts import PromptTemplate

class LLMGCPromptTemplate(PromptTemplate):
    def format(self, data, language="python") -> str:
        return self.template.format(
            name = data.name,
            language = language,
            api = data.api(),
            task_desc = data.task_desc,
            inputs_desc = add_indent(parse_keys(data.inputs_desc)),
            outputs_desc = add_indent(parse_keys(data.outputs_desc)),
            tools_desc = add_indent(parse_keys(data.tools_desc)),
            examples_desc = add_indent(parse_examples(data.examples)),
        )

LLMGC_PROMPT_TEMPLATE = LLMGCPromptTemplate.from_template(
    "Please write a {language} function `{api}` that completes the following goal:\n"
    "{task_desc}\n"
    "The input contains the following keys:\n"
    "{inputs_desc}\n"
    "The output should be returned as a dictionary containing the following values:\n"
    "{outputs_desc}\n"
    "Hint: when implementing the function, you are allowed to use the following tools. You may freely choose to use none, one, many, or all of the tools.\n"
    "Tools:\n"
    "{tools_desc}\n"
    "Examples:\n"
    "{examples_desc}\n"
    "Please respond with the {language} implementation of the `{api}` only. Please do not output any other responses or any explanations.\n"
    "Your respond should be in the following format (the markdown format string should be included):\n"
    "```{language}\n"
    "def {api}:\n"
    "    '''Your Implementation Here.'''\n"
    "```\n"
)

@LinguaManga.register
class LLMGCModule(Module):
    __type__: str = 'module-llmgc'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        if self.task_desc is None: self.task_desc = ""
        if self.inputs_desc is None: self.inputs_desc = dict()
        if self.outputs_desc is None: self.outputs_desc = dict()
        if self.tools_desc is None: self.tools_desc = dict()
        self.inputs = list(self.inputs_desc.keys())
        self.outputs = list(self.outputs_desc.keys())

    def __compile__(self):
        messages = [{"role": "user", "content": LLMGC_PROMPT_TEMPLATE.format(data=self)}]
        return LLMGC(messages=messages)