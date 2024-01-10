import time

import openai

from colorama import Fore, Style
from shorten_paper.logs import logger
from shorten_paper.config import Config

CFG = Config()


def create_chat_completion(
        messages: list,
        lang_model: str = CFG.lang_model_name,
        temperature: float = CFG.model_temperature,
        top_p: float = CFG.model_top_p,
        presence_penalty: float = CFG.model_presence_penalty,
        frequency_penalty: float = CFG.model_frequency_penalty
):
    response = None
    num_retries = 10
    warned_user = False

    for try_num in range(num_retries):
        delay = 5 * (try_num + 1)
        try:
            response = openai.ChatCompletion.create(
                model=lang_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            break
        except openai.error.RateLimitError:
            if warned_user:
                logger.double_check(
                    f"You've got {Fore.YELLOW + Style.BRIGHT}RateLimitError{Style.RESET_ALL} "
                    f"in try number {try_num+1}/{num_retries}."
                )
            else:
                logger.double_check(
                    f"You've got {Fore.YELLOW + Style.BRIGHT}RateLimitError{Style.RESET_ALL} "
                    f"in try number {try_num+1}/{num_retries}."
                    f"Please double check that you have setup a "
                    f"{Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. "
                    f"You can read more here: "
                    f"{Fore.CYAN}https://github.com/Significant-Gravitas/Auto-GPT#openai-api-keys-configuration"
                    f"{Fore.RESET}"
                )
            warned_user = True

        except openai.error.APIError as e:
            if e.http_status == 502:
                pass
            else:
                logger.error(
                    "FAILED TO GET RESPONSE FROM OPENAI",
                    Fore.RED,
                    f"Shorten Paper has failed to get a response from OpenAI's services. "
                    f"Try running Shorten Paper again, "
                    f"and if the problem the persists check your "
                    f"{Fore.CYAN}environment settings{Fore.RESET} and "
                    f"{Fore.CYAN}OpenAI API Account{Fore.RESET}.",
                )
                quit(1)
            if try_num == num_retries - 1:
                logger.error(
                    "FAILED TO GET RESPONSE FROM OPENAI",
                    Fore.RED,
                    f"Shorten Paper has failed to get a response from OpenAI's services. "
                    f"Try running Shorten Paper again, "
                    f"and if the problem the persists check your "
                    f"{Fore.CYAN}environment settings{Fore.RESET} and "
                    f"{Fore.CYAN}OpenAI API Account{Fore.RESET}.",
                )
                quit(1)

        time.sleep(delay)

    if response is None:
        logger.error(
            "FAILED TO GET RESPONSE FROM OPENAI",
            Fore.RED,
            f"Shorten Paper has failed to get a response from OpenAI's services. "
            f"Try running Shorten Paper again, "
            f"and if the problem the persists check your "
            f"{Fore.CYAN}environment settings{Fore.RESET} and "
            f"{Fore.CYAN}OpenAI API Account{Fore.RESET}.",
        )
        quit(1)

    return response.choices[0].message["content"]
