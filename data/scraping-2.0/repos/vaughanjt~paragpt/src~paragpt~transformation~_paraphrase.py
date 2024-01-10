from typing import List, Dict
from triple_quote_clean import TripleQuoteCleaner

tqc = TripleQuoteCleaner()


class StatelessPromptBuilder:
    def __init__(self, prompt: str, remove_new_lines=False):
        if remove_new_lines:
            prompt = prompt.replace("\n", " ")
        self.prompt = prompt

    def __call__(self, text: str) -> str:
        content = "%s\n\n\n%s" % (self.prompt, text)
        return content


prompt_conversation_paraphrase = StatelessPromptBuilder(
    tqc
    << """paraphrase the dialogue of each individual in the following conversation,
    ensure that you respond using the same conversation format e.g.,

        name_1: paraphrase
        name_2: paraphrase
        .
        .
        .
        name_n: paraphrase
        etc....

    maintain each persons full name.
    maintain as much of the key details as possible (e.g. table names, dates, acronyms...),
    remove any filler words,
    fix grammar errors,
    minimize any changes to the content of the conversation,
    focus on requirements, todos and salient information where possible""",
)


def paraphrase(
    conversation: str,
    *args,
    system_prompt,
    user_prompt_builder: "StatelessPromptBuilder" = prompt_conversation_paraphrase,
    model: str = "gpt-3.5-turbo",
    train_of_thought: List[Dict] = None,
    n_retries=5
) -> str:
    import openai
    from openai import APIError
    from paragpt.transformation import no_ssl_verification

    if train_of_thought is None:
        train_of_thought = []

    gpt_input = user_prompt_builder(conversation)

    retry = 1

    while retry <= n_retries:
        try:
            with no_ssl_verification():
                output = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": gpt_input},
                        *train_of_thought,
                    ],
                )
            break
        except APIError as e:
            if retry > n_retries:
                raise e
        retry += 1

    message = output["choices"][0]["message"]["content"]

    return message
