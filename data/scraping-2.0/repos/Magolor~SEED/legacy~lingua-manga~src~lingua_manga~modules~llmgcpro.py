from .default import *
from langchain.prompts import PromptTemplate

class LLMGCProPromptTemplate(PromptTemplate):
    def format(self, data, language="python") -> str:
        return self.template.format(
            name = data.name,
            language = language,
            api = data.api(),
            task_desc = data.task_desc,
            inputs_desc = add_indent(parse_keys(data.inputs_desc)),
            tools_desc = (
                "Hint: when implementing the function, you are allowed to use the following tools. You may freely choose to use none, one, many, or all of the tools.\n" +
                "Tools:\n" +
                add_indent(parse_keys(data.tools_desc))
                if len(data.tools_desc) > 0 else ""
            ),
            examples_desc = add_indent(parse_pro_examples(data.examples[:3])),
        )

LLMGC_PRO_INITIAL_PROMPT_TEMPLATE = LLMGCProPromptTemplate.from_template(
    "Please write a {language} function `{api}` that completes the following goal:\n"
    "{task_desc}\n"
    "The input contains the following keys:\n"
    "{inputs_desc}\n"
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
class LLMGCProModule(Module):
    __type__: str = 'module-llmgcpro'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        if self.task_desc is None: self.task_desc = ""
        if self.inputs_desc is None: self.inputs_desc = dict()
        if self.outputs_desc is None: self.outputs_desc = dict()
        if self.tools_desc is None: self.tools_desc = dict()
        self.inputs = list(self.inputs_desc.keys())
        self.outputs = list(self.outputs_desc.keys())

    def __compile__(self):
        messages = [{"role": "user", "content": LLMGC_PRO_INITIAL_PROMPT_TEMPLATE.format(data=self)}]
        init_cell = LLMGC(messages=messages)
        if self.validator is not None:
            return self.validator.validate(init_cell=init_cell, module=self, prev_cells=self.prev_cells)
        else:
            return init_cell