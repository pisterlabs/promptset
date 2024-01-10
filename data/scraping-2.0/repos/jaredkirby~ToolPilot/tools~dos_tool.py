from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class DosTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Discipline of Study",
            model="gpt-3.5-turbo",
            temperature=0.75,
            uploads=None,
            inputs=[
                {
                    "input_label": "Topic or Objective",
                    "example": "Youtube video writer and editor for a DIY Entrepreneurs focused channel",
                    "button_label": "Discipline",
                    "help_label": 'This tool helps you generate a prompt intro that focus the language model on a particular area of its training. Example: If you want to ask a question about the benefits of exercise, you might use this tool by giving the topics "exercise" and "health".',
                }
            ],
        )

    def execute(self, chat, inputs):
        sys_template = """\
        You are a PhD student who is trying to figure out what discipline of study would best prepare you to answer a question or perform a task based on a topic.
        """
        dos_gen_template = f"""\
            
        What is the discipline of study that would best prepare someone to answer a question
        or perform a task based on the following topic:

        "{inputs}" 

        Format your response as a sentence, including the title of the discipline of study 
        and its correct application to the posed question or task.
        
        Example: "You are [a discipline of study] [relevant application to request]…"
            """
        sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
        user_prompt = HumanMessagePromptTemplate.from_template(dos_gen_template)
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt, sys_prompt])
        formatted_prompt = chat_prompt.format_prompt(user_input=inputs).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
