from langchain.prompts import StringPromptTemplate

system_prompt_template_v1= """You are a writer who has written a chapter wise outline for a topic. You will be given the topic, the current chapter you are working on, the section in the current chapter that you need to summarize, the place of the section in the current chapter, the summary of the previous section (if you're working on the first section in the chapter then the last section from the previous chapter will be provided), the name of the next section and all the contents of a document in between three back ticks(```). You need to write a few paragraphs long summary for the given section in the chapter such that fits perfectly in the complete summary."""

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