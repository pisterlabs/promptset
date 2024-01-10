from langchain.prompts import StringPromptTemplate

human_prompt_template_v1 = """You are writing a blog post on '{topic}'. You are currently writing the section '{section_name}' in which you want to cover {ideas} ideas. You should rewrite a combined summary such it has the main idea, all the supporting points, starts with a thesis statement, is objective, maintains a logical flow and is accurate. The previous section in the blog post is '{previous_section_name}' and the next section is '{next_section_name}'.

```Summaries
{summaries}
```"""
input_variables_v1 = [
    "topic",
    "section_name",
    "ideas",
    "previous_section_name",
    "next_section_name",
    "summaries"
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