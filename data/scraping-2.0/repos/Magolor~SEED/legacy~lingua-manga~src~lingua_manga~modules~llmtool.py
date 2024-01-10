from .default import *
from langchain.prompts import PromptTemplate

class LLMToolPromptTemplate(PromptTemplate):
    def format(self, data) -> str:
        return self.template.format(
            task_desc = data.task_desc,
            tools_desc = add_indent("\n".join([desc for api, desc in data.tools_desc.items()])),
            examples_desc = add_indent(parse_interaction_examples(data.examples)),
            instance = "{instance}",
        )

LLMTOOL_PROMPT_TEMPALTE = LLMToolPromptTemplate.from_template(
    "{task_desc}\n"
    "Please do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:\n"
    "{tools_desc}\n"
    "Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.\n"
    "Interaction examples:\n"
    "{examples_desc}\n"
    "Now consider the following instance:\n"
    "{instance}\n"
)

@LinguaManga.register
class LLMToolModule(Module):
    __type__: str = 'module-llmtool'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        if self.task_desc is None: self.task_desc = ""
        if self.inputs_desc is None: self.inputs_desc = dict()
        if self.outputs_desc is None: self.outputs_desc = dict()
        if self.tools_desc is None: self.tools_desc = dict()
        self.inputs = list(self.inputs_desc.keys())
        self.outputs = list(self.outputs_desc.keys())

    def __compile__(self):
        prompt = LLMTOOL_PROMPT_TEMPALTE.format(data=self)
        build_example_code = "inputs = dict()\n"
        for key in self.inputs:
            build_example_code += f"{parameterize(key, param='inputs')} = {key}\n"
        # for key in self.outputs:
        #     build_example_code += f"{parameterize(key, param='outputs')} = '?'\n"
        input_parser_code = (
            f"def {self.name}_input_parser{self.arguments()}:\n"+ 
            add_indent(self.input_parser_code)+"\n"
        )
        
        code = (
            f"def {self.api()}:\n"+ 
            add_indent(f"'''")+"\n"+
            add_indent(prompt)+"\n"+
            add_indent(f"'''")+"\n"+
            add_indent(build_example_code)+"\n"+
            add_indent(f"instance_prompt = {self.name}_input_parser(**inputs)")+"\n"+
            add_indent(f"prompt = {repr(prompt)}.format(instance=instance_prompt)")+"\n"+
            add_indent(f"return LLMToolExec(prompt)")+"\n"
        )
        
        return Cell(code="\n".join([input_parser_code,code]))