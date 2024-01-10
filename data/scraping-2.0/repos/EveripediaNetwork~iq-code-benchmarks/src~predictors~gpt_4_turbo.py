from langchain.chat_models import ChatOpenAI


def gpt_4_turbo(prompt: str, **kwargs):
    """
    Run the GPT-4 Turbo LLM with the given prompt.
    """
    model = ChatOpenAI(model="gpt-4-1106-preview", **kwargs)
    return model.predict(prompt)
