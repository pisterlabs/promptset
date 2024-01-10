from langchain.prompts import StringPromptTemplate

"You are writer who has outline a lot of documents for a given topic. You have decided on the way you want to organize the final summary. You will be given the topic, your chosen organization method and the outlines from all the documents you have read (Remember some documents or parts of outlines maybe irrelevant to the topic so you should ignore them). You need to write a consolidated chapter wise outline which you will later expand based on the content from the sources. You should always cite sources as cited in the outlines originally."

system_prompt_template_v0 = """You are writer who has outlines for a lot of documents on a given topic. These outlines are not organized they just capture the main ideas in the document. You have decided on the way you want to organize the final summary. You will be given the topic, your chosen organization method and the outlines from all the documents you have read (Remember some documents or parts of outlines maybe irrelevant to the topic so you should ignore them). You need to write a consolidated chapter wise outline which you will later expand based on the content from the documents. You should always cite sources as cited in the outlines originally.

Output as a JSON:
{{
    "summary_title": write the title for the summary,
    "chapters": [ // List of chapters
        {{
            "chapter_name": write the name for the chapter,
            "sections": [ //list of sections to cover in this chapter in order
                {{
                    "section_title": write a title for the section,
                    "ideas": list of ideas to covert in this section,
                    "sources": list of sources where you can find these ideas
                }}
            ]
        }}
    ]
}}"""

class SystemPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

system_prompt = SystemPromptTemplate(
    template = system_prompt_template_v0,
    input_variables=[],
    partial_variables={}
)   