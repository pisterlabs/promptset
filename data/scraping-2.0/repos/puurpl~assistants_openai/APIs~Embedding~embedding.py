from openai import OpenAI


async def create_embedding(model, input, encoding_format):
    client = OpenAI()
    embedding = await client.embeddings.create(
        model=model,
        input=input,
        encoding_format=encoding_format,
    )
    return(embedding)


