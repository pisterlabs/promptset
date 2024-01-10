from langchain.prompts import StringPromptTemplate

human_prompt_template_v1 = """The topic that you have chosen is '{topic}'. You're working on the {section_number} section of '{chapter_name}' which is the {chapter_number} chapter in the summary. The name of this section is '{section_name}'. You want to cover {ideas} in this section. 

```
previous_section_title: {previous_section_title}
previous_section_content: {previous_section_content}
next_section_title: {next_section_title}
content: {content}
```"""

input_variables_v1 = [
    "topic", 
    "section_number",
    "chapter_name",
    "chapter_number",
    "section_name",
    "ideas",
    "previous_section_title",
    "previous_section_content",
    "next_section_title",
    "content"
]

class HumanPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

human_prompt = HumanPromptTemplate(
    template = human_prompt_template_v1,
    input_variables = input_variables_v1
)
