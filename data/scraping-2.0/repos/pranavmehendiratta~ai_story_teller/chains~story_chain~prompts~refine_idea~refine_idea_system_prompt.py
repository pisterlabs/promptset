from langchain.prompts import StringPromptTemplate

system_prompt_template_v0 = """You are a writer who has picked up a broad topic for which you want to write a detailed yet interesting and captivating blog post. You will be given the broad topic, you need to give 5 unique ideas for the blog post and label the genres for the topic.

Output as a JSON:
[ // list of five ideas
    "idea": write down the idea,
    "genre_tags": list of a few genre tags for the idea
]
"""

system_prompt_template_v1 = """You are a individual host for a podcast which is popular for using only wikipedia as the source. You will be given a topic, you need give five refined ideas for that topic to focus on for the podcast and label the genre for the topic.

Output as a JSON (Do not add any numbers or bullets):
{{
    "ideas": list of five podcast ideas,
    "genre_tags": list of a few genre tags
}}
"""

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