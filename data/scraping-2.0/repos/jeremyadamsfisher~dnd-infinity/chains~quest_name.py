from langchain import LLMChain, OpenAI, PromptTemplate

prompt = """My dungeon master just said the following:

> {description}

What is a good name for this quest? Do not explain yourself."""


def generate_quest_name(dialog_turn):
    """Generate a quest name based on the dialog turn."""
    chat_prompt = PromptTemplate(
        input_variables=["description"],
        template=prompt,
    )
    chain = LLMChain(
        llm=OpenAI(temperature=1.0),
        prompt=chat_prompt,
        verbose=True,
    )
    return (
        chain.predict(description=dialog_turn)
        .strip()
        .replace('"', "")
        .replace(".", "")
        .title()
    )
