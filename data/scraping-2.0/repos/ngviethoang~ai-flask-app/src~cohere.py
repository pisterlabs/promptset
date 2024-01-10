import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def summarize(
    text: str,
    temperature: float = None,
    length: str = "medium",
    format: str = None,
):
    return co.summarize(
        text=text,
        temperature=temperature,
        model="summarize-xlarge",
        length=length,
        format=format,
    )
