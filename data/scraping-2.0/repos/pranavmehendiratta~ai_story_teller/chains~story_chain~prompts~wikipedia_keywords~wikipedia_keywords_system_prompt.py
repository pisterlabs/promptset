from langchain.prompts import StringPromptTemplate

system_prompt_template_v1 = """You are a individual host for a podcast which is popular for using only wikipedia as the source. You already have an idea for the podcast. Break that idea down into keywords which can used to research on the topic on wikipedia.

Output the ideas as follows:
{{
    "wikipedia_keywords": list of keywords to search on wikipedia
}}
"""

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