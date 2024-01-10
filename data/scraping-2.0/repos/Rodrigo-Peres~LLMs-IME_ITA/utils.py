import json
import numpy as np
import openai
import pandas as pd
import random
import tiktoken
import time

from anthropic import Anthropic
from vertexai.preview.language_models import (
    ChatModel,
    TextGenerationModel,
)


QUESTIONS_FILE_PATH = "data/IME_and_ITA_questions_dataset.xlsx"

MAX_RETRIES = 10
INITIAL_DELAY = 1
BACKOFF_FACTOR = 2

MAX_TOKENS = 4096


with open("./credentials/ANTHROPIC_KEY.json") as f:
    anthropic_key = json.load(f)

with open("./credentials/GPT_SECRET_KEY.json") as f:
    open_ai = json.load(f)
openai.api_key = open_ai["API_KEY"]


def tokens_num_allowed(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    tokens_num_allowed = MAX_TOKENS - num_tokens
    return tokens_num_allowed


def prep_dataset(dataframe_path):
    if hasattr(prep_dataset, "cached_df"):
        return prep_dataset.cached_df
    df = pd.read_excel(dataframe_path)
    df["extra_info"].replace(np.nan, None, inplace=True)
    df = df[df["status"] == "OK"]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    prep_dataset.cached_df = df
    return df


def handle_errors(retries, max_retries):
    if retries == max_retries - 1:
        raise Exception("Max retries exceeded.")
    delay = INITIAL_DELAY * (BACKOFF_FACTOR ** retries)
    delay += random.randint(0, delay)
    time.sleep(delay)


def get_prompt(row):
    prompt = f"Questão: {row['question']}\n\n"
    if row["extra_info"]:
        prompt += f"Considere essas informações adicionais:\n\n{row['extra_info']}\n\n"
    prompt += f"Alternativas:\n\n{row['alternatives']}\n\nResposta: "
    return prompt


def get_answer(prompt_technique, model, row, temperature=0):
    retries = 0
    df_questions = prep_dataset(QUESTIONS_FILE_PATH)

    while retries < MAX_RETRIES:
        print(f"Retries num {retries}")

        try:
            if prompt_technique == "zero_shot":
                prompt = get_prompt(row)

            if prompt_technique == "zero_shot_chain_of_thought":
                zero_shot_cot_prompt = (
                    "Vamos pensar passo a passo para resolver a questão."
                )
                prompt = get_prompt(row) + zero_shot_cot_prompt

            if prompt_technique == "plan_and_solve":
                plan_and_solve_prompt = "Vamos primeiro entender o problema, extrair variáveis relevantes, seus números correspondentes e fazer um plano. Então, vamos executar o plano, calcular as variáveis intermediárias (atenção ao cálculo numérico correto e ao bom senso), resolver o problema passo a passo e mostrar a resposta."
                prompt = get_prompt(row) + plan_and_solve_prompt

            if prompt_technique in ["few_shot", "chain_of_thought"]:
                prompt = get_prompt(row)
                df_examples = df_questions[
                    (df_questions["exam"] == row["exam"])
                    & (df_questions["year"] != row["year"])
                    & (df_questions["subject"] == row["subject"])
                    & (df_questions[prompt_technique].notnull())
                ]
                prompt_list = []

                for _, row in df_examples.iterrows():
                    new_prompt = get_prompt(row) + row[prompt_technique] + "\n"
                    prompt_list.append(new_prompt)

                prompt = "\n".join(prompt_list + [prompt])

            print(prompt)
            if model in ["text-davinci-003", "gpt-3.5-turbo-0613", "gpt-4-0613"]:
                tokens_allowed = tokens_num_allowed(prompt, model)
                print(f"Tokens left: {tokens_allowed}")

            if model == "text-davinci-003":
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    presence_penalty=0,
                    frequency_penalty=0,
                    max_tokens=tokens_allowed,
                )
                return response.choices[0].text.strip()

            if model in ["gpt-3.5-turbo-0613", "gpt-4-0613"]:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente muito capacitado em resolver questões complexas de Matemática, Química, Física e Português.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    presence_penalty=0,
                    frequency_penalty=0,
                    max_tokens=tokens_allowed - 42,
                )
                return response.choices[0].message["content"].strip()

            if model == "text-bison@001":
                modelo_instanciado = TextGenerationModel.from_pretrained(model)
                response = modelo_instanciado.predict(
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=1024,
                    top_k=1,
                    top_p=0.8,
                )
                return response.text.strip()

            if model == "chat-bison@001":
                chat_model = ChatModel.from_pretrained(model)
                chat = chat_model.start_chat(
                    context="Você é um assistente muito capacitado em resolver questões complexas de Matemática, Química, Física e Português.",
                )
                response = chat.send_message(
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=1024,
                    top_k=1,
                    top_p=0.8,
                )
                return response.text.strip()

            if model in ["claude-instant-1.1-100k", "claude-1.3"]:
                anthropic = Anthropic(api_key=anthropic_key)
                response = anthropic.completions.create(
                    model=model,
                    max_tokens_to_sample=100000,
                    stop_sequences=None,
                    temperature=temperature,
                    prompt=prompt,
                )
                return response.completion.strip()

        except Exception as e:
            handle_errors(retries, MAX_RETRIES)
            retries += 1


def extract_right_option(answer):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            prompt = f"""
            Extraia do texto somente a letra que representa a resposta correta.\n
            Texto: Resposta: d) [18001, 19000].\n
            D\n\n
            Texto: A resposta correta é a letra e) XeF_3^+, SF_4, ClF_3.\n
            E\n\n
            Texto: C) cumprem a função de destacar o absurd\n
            C\n\n
            Texto: {answer}\n
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message["content"].strip()

        except Exception as e:
            handle_errors(retries, MAX_RETRIES)
            retries += 1
