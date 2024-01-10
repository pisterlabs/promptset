from langchain.prompts import StringPromptTemplate

system_prompt_template_v1_gpt_4 = """You are writing a script for a podcast. You already have a style, tone, narration, structure and opening for the script. You'll be given the opening for the current section, topics to cover in current section, topics to cover in the next section (to help you write the opening for the next section). You need to write a detailed yet engaging current section and opening for the next section.

Podcast Information:
num_of_hosts: you are the only host
num_of_podcasts: this topic needs to be covered in one podcast episode of about 20 mins long

Output only json object (I will be parsing this in python always escape characters where necessary!):
{{
    "section_content": write the complete section,
    "next_section_opening": write the opening for the next section
}}
"""

system_prompt_template_v1_gpt_3_5 = """You are writing a script for a podcast. You already have a style, tone, narration, structure and opening for the script. You'll be given the opening for the current section, topics to cover in current section, topics to cover in the next section (to help you write the opening for the next section) and a context. You need to write a detailed yet engaging current section and opening for the next sectio only using the context.

Podcast Information:
num_of_hosts: you are the only host
num_of_podcasts: this topic needs to be covered in one podcast episode of about 20 mins long

Output only json object (this will be parsed in python. Use newlines correctly!!):
{{
    "section_content": write the complete section,
    "next_section_opening": write the opening for the next section
}}
"""

class SystemPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

system_prompt_gpt_4 = SystemPromptTemplate(
    template = system_prompt_template_v1_gpt_4,
    input_variables=[],
    partial_variables={}
)   

system_prompt = SystemPromptTemplate(
    template = system_prompt_template_v1_gpt_3_5,
    input_variables=[],
    partial_variables={}
)