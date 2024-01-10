#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import openai
import tiktoken
from csv import DictWriter
from huggingface_hub import Repository
from datetime import datetime
import os
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")

repo_url = (
    "https://huggingface.co/datasets/kdercksen/" "medical-patient-chatbot-conversations"
)
data_file = Path(repo_url) / "data" / "data.csv"
HF_TOKEN = os.environ.get("HF_TOKEN")
repo = Repository(local_dir="data", clone_from=repo_url, use_auth_token=HF_TOKEN)


def store_conversation(messages):
    with open(data_file, "a") as csvfile:
        writer = DictWriter(csvfile, fieldnames=["messages", "timestamp"])
        writer.writerow({"messages": messages, "timestamp": str(datetime.now())})
        commit_url = repo.push_to_hub()


# Collect information on the various examinations available
examinations = {}
for fname in sorted(Path("./onderzoeken").glob("*.txt")):
    with open(fname) as f:
        examinations[fname.stem] = f.read()

pricing = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
}


def make_system_prompt(exam):
    intro = (
        "Je bent een assistent bij het ziekenhuis RadboudUMC. "
        "Je helpt patienten met vragen over hun bezoek in "
        "hun eigen taal en op hun eigen niveau."
    )
    exam_specific = (
        "Beantwoord alle vragen met behulp van de volgende informatie. "
        "Als het antwoord niet in de tekst staat, zeg dat dan expliciet "
        f"in plaats van zelf een antwoord te bedenken.\n\n{examinations[exam]}"
    )
    return intro + exam_specific


def format_chat_history(chat_history, exam):
    system_prompt = make_system_prompt(exam)
    messages = [{"role": "system", "content": system_prompt}]
    for pair in chat_history:
        if pair[0] is not None:
            messages.append({"role": "user", "content": pair[0]})
        if pair[1] is not None:
            messages.append({"role": "assistant", "content": pair[1]})
    return messages


with gr.Blocks() as demo:
    with gr.Row():
        api_key = gr.Textbox(label="Enter your OpenAI API key here")

        model_choice = gr.Dropdown(
            label="Choose the model to use for the conversation",
            choices=(c := sorted(pricing.keys())),
            value=c[0],
        )

    with gr.Row():
        exam = gr.Dropdown(
            label="Examination", choices=(c := sorted(examinations.keys())), value=c[0]
        )
        total_cost = gr.Markdown("## Total cost of conversation: $0.00")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])
        save_conversation = gr.Button("Save conversation")
    tokens_used = gr.State(value={"prompt": 0, "completion": 0})

    def update_user_msg(message, chat_history):
        return "", chat_history + [[message, None]]

    def respond(chat_history, exam, model_choice, api_key):
        response = openai.ChatCompletion.create(
            api_key=api_key,
            model=model_choice,
            messages=format_chat_history(chat_history, exam),
            stream=True,
        )
        chat_history[-1][1] = ""
        for chunk in response:
            if "content" in chunk.choices[0].delta:
                chat_history[-1][1] += chunk.choices[0].delta.content
                yield chat_history

    def update_cost(chat_history, exam, tokens_used):
        system_prompt_tokens = len(enc.encode(make_system_prompt(exam)))
        user_messages_tokens = sum([len(enc.encode(pair[0])) for pair in chat_history])
        assistant_messages_tokens = sum(
            [len(enc.encode(pair[1])) for pair in chat_history]
        )
        return {
            "prompt": tokens_used["prompt"]
            + system_prompt_tokens
            + user_messages_tokens,
            "completion": tokens_used["completion"] + assistant_messages_tokens,
        }

    def render_cost(tokens_used, model_choice):
        prompt_price = tokens_used["prompt"] / 1000 * pricing[model_choice]["prompt"]
        completion_price = (
            tokens_used["completion"] / 1000 * pricing[model_choice]["completion"]
        )
        return f"## Total cost of conversation: ${prompt_price+completion_price:.2f}"

    # Update user message, then stream output, then update used tokens once streaming is
    # done, then render the new cost to markdown.
    msg.submit(update_user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(
        respond, [chatbot, exam, model_choice, api_key], chatbot
    ).then(update_cost, [chatbot, exam, tokens_used], tokens_used).then(
        render_cost, [tokens_used, model_choice], total_cost
    )
    clear.click(
        lambda: (None, {"prompt": 0, "completion": 0}),
        None,
        [chatbot, tokens_used],
        queue=False,
    ).then(render_cost, [tokens_used, model_choice], total_cost)

if __name__ == "__main__":
    demo.queue()
    demo.launch()