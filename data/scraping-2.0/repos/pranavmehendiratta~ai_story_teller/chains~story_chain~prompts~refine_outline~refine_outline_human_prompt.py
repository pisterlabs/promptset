from langchain.prompts import StringPromptTemplate

human_prompt_template_v1 = """Below are a set of outlines that you need to combine into a single outline which you will later use to write a script for a captivating podcast.

```
{outlines}
```"""
input_variables_v1 = ["outlines"]

class HumanPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

human_prompt = HumanPromptTemplate(
    template = human_prompt_template_v1,
    input_variables = input_variables_v1
)