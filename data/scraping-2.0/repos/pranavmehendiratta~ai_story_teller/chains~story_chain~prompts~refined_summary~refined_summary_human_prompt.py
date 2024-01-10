from langchain.prompts import StringPromptTemplate

human_prompt_template_v1 = """Refine the summary given below in between three back ticks (```) such that after refining it still contains all the valuable information but the content is clear, precise and has a logical flow to it.

```
{summary}
```"""
input_variables_v1 = [
    "summary"
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