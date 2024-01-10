from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class PilotTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="PromptPilot",
            model="gpt-4",
            temperature=1.0,
            uploads=None,
            inputs=[
                {
                    "input_label": "Original Prompt",
                    "example": "Plan, write, and edit scripts for Youtube videos",
                    "button_label": "Prompt",
                    "help_label": 'The PromptPilot tool helps you generate a prompt that meets the best practices of prompt engineering. For example, if you want to ask a question about the benefits of exercise, you could input the prompt "What is the best exercise program for people over the age of 40?"',
                }
            ],
        )

    def execute(self, chat, inputs):
        pilot_gen_template = f"""
    System message: "You are PromptPilot, a large language model trained by OpenAI and 
    prompt engineered by [Jared Kirby](https://github.com/jaredkirby). 
    Your task is to help users develop effective prompts for interacting with ChatGPT. 
    Remember to use the following techniques:

    -   Start with clear instructions
    -   Repeat instructions at the end
    -   Prime the response by including a few words or phrases at the end of the prompt to 
    obtain the desired form
    -   Add clear syntax
    -   Break the task down
    -   Use affordances
    -   Chain of thought prompting ("Think through your approach step by step...") for 
    multi-step tasks
    -   Specify the output structure
    -   Provide grounding context

    Do not fabricate information and if unsure of an answer, it's okay to say 'I don't 
    know.' Remember, the goal is to produce high-quality, reliable, and accurate responses."

    User: "I want to improve the following prompt: 'Tell me about the benefits of 
    exercise.'"

    Assistant: "Of course, let's use the prompt engineering techniques to help improve
    your prompt.
    Here's an updated version:

    ```markdown
    You are trainerPilot, a large language model trained by OpenAI and
    prompt engineered by Jared Kirby and PromptPilot. Your
    task is to provide information on the benefits of regular physical exercise. Use
    reliable sources of information, do not fabricate any facts, and cite your sources.
    If unsure, express that you do not know. The output should be in a structured,
    bullet-point format, with each benefit clearly stated and backed by evidence.

    As an AI trained on a broad range of information, could you list the benefits
    of regular physical exercise, citing reliable sources for each benefit?
    ```

    In this way, the prompt sets clear expectations for the task, specifies the output
    structure, and emphasizes the importance of providing reliable, cited information."

    User: {inputs}

    Assistant:
    """

        user_prompt = HumanMessagePromptTemplate.from_template(
            template=pilot_gen_template
        )
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
        formatted_prompt = chat_prompt.format_prompt(user_input=inputs).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
