from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

def chat_prompt_template(chat_model, prompt_inject, *, system_message_template_inject, human_message_example_inject, ai_message_example_inject):
    
    system_message_template = ("""{system_message_template_inject}""")   
    human_message_example = ("""{human_message_template_inject}""")
    ai_message_example = ("""{ai_message_template_inject}""")
    human_message_template = ("""{prompt_inject}""")

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template, system_message_template_inject=system_message_template_inject)
    human_message_example = SystemMessagePromptTemplate.from_template(human_message_example, additional_kwargs={"name": "example_user"}, human_message_template_inject=human_message_example_inject)
    ai_message_example = SystemMessagePromptTemplate.from_template(ai_message_example, additional_kwargs={"name": "example_ai"}, ai_message_template_inject=ai_message_example_inject)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template, prompt_inject=prompt_inject)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_example, ai_message_example, human_message_prompt]
    )

    results = chat_model(
        chat_prompt.format_prompt(
        system_message_template_inject=system_message_template_inject,
        human_message_template_inject=human_message_example_inject,
        ai_message_template_inject=ai_message_example_inject,
        prompt_inject=prompt_inject,
        ).to_messages()
    )

    return results.content