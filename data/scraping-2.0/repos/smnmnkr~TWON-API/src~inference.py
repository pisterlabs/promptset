from huggingface_hub import InferenceClient
from openai import OpenAI

from .schemas import Integration


def inference(
        template: str,
        variables: dict,
        integration: Integration
) -> dict:
    prompt: str = template.format(**variables)

    return {
        'prompt': prompt,
        'response': {
            'huggingFace': hf_inference,
            'OpenAI': oai_inference,
        }[integration.provider](
            prompt,
            model=integration.model
        ).strip('\n').strip(),
    }


def hf_inference(prompt: str, model: str) -> str:
    return (
        InferenceClient(model)
        .text_generation(
            prompt,
            max_new_tokens=250
        )
    )


def oai_inference(prompt: str, model: str) -> str:
    return (
        OpenAI()
        .chat
        .completions
        .create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model
        )
        .choices[0]
        .message
        .content
    )
