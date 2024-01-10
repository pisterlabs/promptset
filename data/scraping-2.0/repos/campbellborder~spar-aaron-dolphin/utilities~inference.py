from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
from tqdm import tqdm
import openai
import time
from dotenv import load_dotenv
import os
import requests
import together

load_dotenv()

def get_openai_chat_messages(
    system_prompt: str,
    few_shot_prompts: list[tuple[str, str]],
    question: str,
    ai_message_prompt: str,
    ) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for prompt in few_shot_prompts:
        q, a = prompt
        messages.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ])
    messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": ai_message_prompt}
    ])
    return messages


def openai_chat_inference(
    full_model_name: str,
    system_prompt: str,
    few_shot_prompts: list[tuple[str, str]],
    dataset: list[dict[str, str]],
    max_tokens: int,
    temperature: int,
    include_ai_message_prompt: bool = True,
    ) -> list[dict[str, str]]:
    outputs = []
    for item in tqdm(dataset):
        time.sleep(0.5)
        question = item["question"]
        answer_matching_behavior = item["answer_matching_behavior"]
        answer_not_matching_behavior = item["answer_not_matching_behavior"]

        ai_message_prompt = "My answer is (" if include_ai_message_prompt else ""
        messages = get_openai_chat_messages(system_prompt, few_shot_prompts, question, ai_message_prompt)

        response = openai.ChatCompletion.create(
            model=full_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        model_output = response.choices[0].message
        model_answer = f"{ai_message_prompt} {model_output.content}"

        output = {
            "question": question,
            "answer_matching_behavior": answer_matching_behavior,
            "answer_not_matching_behavior": answer_not_matching_behavior,
            "model_answer": model_answer,
        }
        outputs.append(output)
    return outputs


def get_llama_chat_messages(
    system_prompt: str,
    few_shot_prompts: list[tuple[str, str]],
    ) -> str:
    """
    Format the few-shot message for Llama-2-chat.

    Example format:
    <s> [INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question1} [/INST] {answer1} </s>
    <s> [INST] question2 [/INST] answer2 </s>
    <s> [INST] question3 [/INST] My answer is (
    """
    SOS, EOS = "<s>", "</s>"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    messages = []

    for i, (question, answer) in enumerate(few_shot_prompts):
        message = ""
        if i == 0:
            message = B_SYS + system_prompt + E_SYS + question
        else:
            message = question

        if i < len(few_shot_prompts) - 1:
            message = f"{SOS} {B_INST} {message} {E_INST} {answer} {EOS}"
        else:
            message = f"{SOS} {B_INST} {message} {E_INST} {answer}"

        messages.append(message)
    return "\n".join(messages)


def query(prompt):
    output = together.Complete.create(
        prompt=prompt,
        model = "togethercomputer/llama-2-70b-chat", 
        max_tokens = 10,
        temperature = 1.0,
    )
    text = output['output']['choices'][0]['text']
    return text


def together_inference(
    full_model_name: str,
    system_prompt: str,
    few_shot_prompts: list[tuple[str, str]],
    dataset: list[dict[str, str]],
    max_tokens: int,
    temperature: int,
    include_ai_message_prompt: bool = True,
    ) -> list[dict[str, str]]:
    outputs = []
    for item in tqdm(dataset):
        time.sleep(0.25)
        question = item["question"]
        answer_matching_behavior = item["answer_matching_behavior"]
        answer_not_matching_behavior = item["answer_not_matching_behavior"]

        ai_message_prompt = "My answer is ("
        history = get_llama_chat_messages(
            system_prompt,
            few_shot_prompts + [(question, ai_message_prompt)],
        )

        model_output = query(history)
        model_answer = f"{ai_message_prompt} {model_output}"

        output = {
            "question": question,
            "answer_matching_behavior": answer_matching_behavior,
            "answer_not_matching_behavior": answer_not_matching_behavior,
            "model_answer": model_answer,
        }
        outputs.append(output)
    return outputs
