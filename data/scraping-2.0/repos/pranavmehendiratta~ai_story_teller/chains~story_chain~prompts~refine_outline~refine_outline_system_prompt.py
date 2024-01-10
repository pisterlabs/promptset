from langchain.prompts import StringPromptTemplate

system_prompt_template_v1 = """You are writing a script for a captivating podcast. You have already come with a set outlines based on different articles, books, wikipedia etc that you have read. Before writing the script, you are going to refine the outlines into a single outline.
All outlines are between three backticks (```). Each outline is separated by "{outline_separator}"."""

input_variables_v1 = ["outline_separator"]

class SystemPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

system_prompt = SystemPromptTemplate(
    template = system_prompt_template_v1,
    input_variables = input_variables_v1,
    partial_variables = {}
)   