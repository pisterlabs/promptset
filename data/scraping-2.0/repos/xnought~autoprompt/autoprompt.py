from __future__ import annotations
import openai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def embed_batch(batch, model="text-embedding-ada-002", api_key=None):
    if api_key is not None:
        openai.api_key = api_key

    batch = [text.replace("\n", " ") for text in batch]
    response = openai.Embedding.create(input=batch, model=model)
    embeddings = [d["embedding"] for d in response["data"]]
    return np.array(embeddings)


def top_similar(prompt_embedding, db, k=10):
    norms = np.linalg.norm(db - prompt_embedding, axis=1)
    args = norms.argsort()
    return args[:k]


def chatgpt(
    prompt="Who won the world series in 2020?", model="gpt-3.5-turbo", api_key=None
):
    if api_key is not None:
        openai.api_key = api_key

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def synthesize(historic_prompts, chat_model=lambda x: chatgpt(x, model="gpt-4")):
    template = """
    Given the following prompts were rated high quality

    {}

    Synthesize the best parts of the prompts above and list them from most effective to least effective
    Do not repeat the prompt themselves, just take the patterns that are most effective.
    Do not explain the reason why they are effective, just list them. You can include entire prompts, parts of them, or synthesized parts of them.
    You may want to use a template syntax to make this easier.
    Rank the top most effective patterns first, then the next most effective, and so on.
    Do not just create a long list of the exact prompts, but rather the patterns that make them effective in a potent and concise list.
    Limit the output to the top 10 most effective patterns and the top most used names, places, and things, as well as creative and interesting patterns.
    """
    synth = chat_model(template.format("\n".join(historic_prompts)))
    return synth


def autoprompt(
    prompt: str,
    chat_model=lambda x: chatgpt(x, model="gpt-4"),
    history=[],
) -> str:
    """autoprompt
    Takes in a prompt and completes the prompt. Kind of like github copilot but for prompts.

    the history are past effective prompts you'd wish to get inspiration from
    """
    effective_patterns = synthesize(history, chat_model=chat_model)
    history_template = """Given these prompting pattens were effective (ordered by most helpful first): {}
    Understand and make use of these effective patterns. Take the main ideas and make us of them if they apply and would help the prompt become
    more effective or more interesting and creative.\n"""
    template = """You are an expert prompt engineer and question asker. Complete the following
    prompt by adding words or phrases that would make the prompt more effective. Don't add words or phrases that would
    make the prompt less effective. Use hacks that would make the prompt more effective. You may leverage the history of prompts in your completion.
    Don't repeat concepts contained in the prompt, only add to it.

    Do not to repeat the Prompt the user provides in your completion. Only add to it incrementally.
    Make the additions as short as possible. Don't add more than 5 words. Make it potent, not long winded. To the point and NOT a super long sentence completion.
    Try to only use on new incremental idea in your completion to maximize its usefulness as a question/prompt.
    
    Prompt: {}
    Completion:
    """
    engineered_completion_prompt = (
        history_template.format(effective_patterns) + template.format(prompt)
        if len(history) > 0
        else template.format(prompt)
    )
    return chat_model(engineered_completion_prompt)


def vectordb_autoprompt(
    prompt: str,
    embeddings_history: np.ndarray,
    history: list[str],
    chat_model=lambda x: chatgpt(x, model="gpt-4"),
    embed_model=lambda x: embed_batch(x)[0],
    top_k_similar=10,
):
    prompt_embedding = embed_model(prompt)
    indexes_similar = top_similar(prompt_embedding, embeddings_history, k=top_k_similar)
    collection = [history[i] for i in indexes_similar]
    return autoprompt(prompt, chat_model, collection)
