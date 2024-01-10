from langchain.chat_models import ChatOpenAI


def gpt_3_5_turbo(prompt: str, **kwargs):
    """
    Run the GPT-3.5 Turbo LLM with the given prompt.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", **kwargs)
    return model.predict(prompt)
