from openai import OpenAI


def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    """Embedding using OpenAI API

    Args:
        text (str): full text job description
        model (str, optional): _description_. Defaults to "text-embedding-ada-002".

    Returns:
        _type_: _description_
    """
    client = OpenAI()

    text = text.replace("\n", " ")

    return client.embeddings.create(input=[text], model=model).data[0].embedding
