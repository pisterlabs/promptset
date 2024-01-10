import openai
from decouple import config


def generate_meditation_prompt(length_choice, focus, method):
    openai.api_key = config("OPENAI_API_KEY")

    model = "gpt-3.5-turbo"

    if length_choice == "short":
        description = "around 2-3 minutes"
        interval_duration = "30 seconds"
        intervals = "00:30, 01:00, 01:30, and so on"
    elif length_choice == "medium":
        description = "around 5 minutes"
        interval_duration = "1 minute"
        intervals = "01:00, 02:00, 03:00, and so on"
    elif length_choice == "long":
        description = "around 10 minutes"
        interval_duration = "2 minutes"
        intervals = "02:00, 04:00, 06:00, and so on"
    else:
        raise ValueError("Invalid length_choice provided")

    if method == "none" or not method:
        prompt_text = (
            f"Concisely craft a meditation script for {description} focusing on {focus}. "
            f"Provide instructions with intervals of {interval_duration}. "
            f"Use timestamps that progress as {intervals}. "
            f"Format: '00:00 - txt'. No square brackets or extraneous content or spaces between lines."
        )
    else:
        prompt_text = (
            f"Concisely craft a meditation script for {description} focusing on {focus} employing the {method} technique. "
            f"Provide instructions with intervals of {interval_duration}. "
            f"Use timestamps that progress as {intervals}. "
            f"Format: '00:00 - txt'. Exclude any square brackets or superfluous content or spaces between lines."
        )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
    )

    return response.choices[0].message["content"].strip()
