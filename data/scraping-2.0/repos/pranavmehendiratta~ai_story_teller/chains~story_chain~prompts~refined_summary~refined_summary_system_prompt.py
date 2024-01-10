from langchain.prompts import StringPromptTemplate

system_prompt_template_v1 = """"""

class SystemPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

system_prompt = SystemPromptTemplate(
    template = system_prompt_template_v1,
    input_variables=[],
    partial_variables={}
)   