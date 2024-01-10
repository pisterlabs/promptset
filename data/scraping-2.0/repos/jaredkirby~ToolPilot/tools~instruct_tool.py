from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class InstructTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Improve Prompt Instructions",
            model="gpt-4",
            temperature=0.5,
            uploads=None,
            inputs=[
                {
                    "input_label": "Original Instructions",
                    "example": "Write and edit scripts for Youtube videos",
                    "button_label": "Instructions",
                    "help_label": "The Improve Prompt Instructions tool helps you generate a prompt that provides clear instructions for the language model. For example, if you want to generate a specific exercise program, you could input the instructions'Generate a 12-week exercise program for people over the age of 40 and include a list of exercises, sets, and reps for each day of the week.",
                }
            ],
        )

    def get_instruct_response(self, chat, inputs):
        instruct_gen_template = f"""
    You are a technical communicator/instructional designer evaluating and revising
    instructions to be explicit, specific, and helpful by anticipating and preempting 
    possible failure. Think through the following process step by step to ensure nothing 
    is missed, and no mistakes are made, then respond with your improved revision.

    ---
    {inputs}
    ---

    Please respond with the revised instructions ONLY.
    Do not complete the given process.
    """

        user_prompt = HumanMessagePromptTemplate.from_template(
            template=instruct_gen_template
        )
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
        formatted_prompt = chat_prompt.format_prompt(user_input=inputs).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
