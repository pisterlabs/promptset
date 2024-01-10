from langchain.prompts import StringPromptTemplate

system_prompt_template_v1 = """You are writer who has summarized an idea into separate summaries based on each of the documents that you have read. You now want to combine the all the summaries of the same section into one summary. In the final summary you don't want any sections or chapters so the summary should have logical flow from introduction to ending. You will be given the title of the summary, the ideas you want to cover in that summary, and all the separate summaries in between three back ticks(```). Each summary will be separated by '\n\n-- Next Summary --\n\n'."""

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