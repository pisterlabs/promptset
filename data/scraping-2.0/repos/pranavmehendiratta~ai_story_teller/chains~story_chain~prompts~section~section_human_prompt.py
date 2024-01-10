from langchain.prompts import StringPromptTemplate

human_prompt_template_v1_gpt_4 = """topic = {topic}\n
structure = {structure}\n
style = {style}\n
narration = {narration}\n
tone = {tone}\n
current_section_opening = {current_section_opening}\n
topics_to_cover_in_current_section = {topics_to_cover_in_current_section}\n
topics_to_cover_in_next_section = {topics_to_cover_in_next_section}\n"""

input_variables_v1_gpt_4 = ["topic", "structure", "style", "narration", "tone", "current_section_opening", "topics_to_cover_in_current_section", "topics_to_cover_in_next_section"]

human_prompt_template_v1_gpt_3_5 = """topic = {topic}\n
structure = {structure}\n
style = {style}\n
narration = {narration}\n
tone = {tone}\n
current_section_opening = {current_section_opening}\n
topics_to_cover_in_current_section = {topics_to_cover_in_current_section}\n
topics_to_cover_in_next_section = {topics_to_cover_in_next_section}\n
context: {context}\n"""

input_variables_v1_gpt_3_5 = ["topic", "structure", "style", "narration", "tone", "current_section_opening", "topics_to_cover_in_current_section", "topics_to_cover_in_next_section", "context"]

class HumanPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

human_prompt_gpt_4 = HumanPromptTemplate(
    template = human_prompt_template_v1_gpt_4,
    input_variables = input_variables_v1_gpt_4
)

human_prompt = HumanPromptTemplate(
    template = human_prompt_template_v1_gpt_3_5,
    input_variables = input_variables_v1_gpt_3_5
)
