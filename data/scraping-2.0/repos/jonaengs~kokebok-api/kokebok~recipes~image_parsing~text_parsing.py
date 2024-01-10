import math
from typing import Literal

import openai

ALLOWED_MODELS = Literal["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
DEFAULT_GPT_MODEL: ALLOWED_MODELS = "gpt-3.5-turbo"

# Pricing is USD per 1k tokens
# Pricing data updated: 2023/10/13
GPT_PRICING: dict[ALLOWED_MODELS, dict[Literal["input", "output"], float]] = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-4": {"input": 0.03, "output": 0.06},
}

GPT_SYSTEM_PROMPT = """\
Your task is to structure text. Specifically, you will be given text
extracted from images of cooking recipes, and will give structure to the
text by dividing it into the following categories:
    title, preamble, yield, content, instructions, ingredients.

Each piece of text fits in a single category.
Many of the categories may not appear in the recipe. However,
the "instructions" and "ingredients" categories almost always do.
The "content" category is only to be used for pieces of text that do
not fit in any of the other categories.

Ingredients may be grouped by which part of the recipe they are used in.
Try to preserve these groupings if possible.
In the following example, there are two groups and four ingredients:
"
For the chicken:
1 lb chicken thighs
2 tsp chicken spice
For the sauce:
1/4 cup white wine
1 tbsp unsalted butter
"
Therefore, the groupings will be as follows:
```
{
    "For the chicken": ["1 lb chicken thighs", "2 tsp chicken spice"]`,
    "For the sauce": ["1/4 cup white wine", "1 tbsp unsalted butter"],
}
```

The user input will be the recipe text. Your reply should be the recipe text
divided up into the categories described above.

You are NOT allowed to alter the recipe text in any semantically meaningful way.
You will not duplicate text.
You may remove words and/or characters if it is clear that they are wrongful
artefacts produced by the OCR performed on the image.

The output shoudl be formatted using the JSON format. Your output will be a
single JSON object with a series of keys mapping to values.
Each category will be a key, and the text belonging to that category will
be the value. You may turn strings that represent lists into JSON arrays.

Groupings of ingredients should be preserved. This is achieved by
representing the ingredients as a JSON object, with the keys being
the ingredient group names and the values being the list of ingredients
belonging to that group. If no group name for the ingredients is given,
all ingredients can be placed under a single a key equalling the empty string ("").

An example output object could look like this:
{
    "title": "Pancakes with homemade blueberry jam",
    "ingredients": {
        "Pancakes": [
            "1 packet of pancake mix",
            "Butter",
        ],
        "Blueberry jam": [
            "300 grams fresh blueberries",
            "100 grams sugar"
        ]
    },
    "instructions": [
        "Create the pancake batter as instructed on the packet",
        "Leave the batter to swell",
        "Mix the blueberries and sugar, before crushing them with a fork",
        "Fry the pancakes",
        "Serve the fresh pancakes with your delicious homemade blueberry jam"
    ],
    "yields": "2 servings"
}
"""

# noqa
GPT_USER_HINT_INTRO = (
    "You have been provided with the following information about "
    "the document to help you parse the it correctly:\n"
)


def text_to_recipe(text: str, user_hint: str = "") -> str:
    """
    Asks ChatGPT to structure the input text according to the system prompt.
    Returns ChatGPT's response (assumed to be valid JSON) as a dict.
    """

    # Estimate the number of tokens in the input. Likely a pessimistic estimate
    # System prompt is in English, which generally has 4 tokens per character.
    estimate_system_prompt_tokens = len(GPT_SYSTEM_PROMPT) / 3.5
    # User input may be non-english and contain numeric and special characters.
    # Thus it is likely/possible that char-to-token ratio is lower.
    estimate_user_input_tokens = len(text) / 2.5
    estimate_total_input_tokens = math.ceil(
        estimate_system_prompt_tokens + estimate_user_input_tokens
    )
    print(f"{estimate_total_input_tokens=}")

    # Set GPT model to use
    gpt_model: ALLOWED_MODELS = DEFAULT_GPT_MODEL
    if estimate_total_input_tokens > 16_000:
        raise ValueError("Text input calculated too large for model context.")
    if gpt_model == "gpt-3.5-turbo" and estimate_total_input_tokens > 4_000:
        gpt_model = "gpt-3.5-turbo-16k"

    # Construct the chat messages
    chat_messages = [{"role": "system", "content": GPT_SYSTEM_PROMPT}]
    if user_hint:
        chat_messages.append(
            {"role": "system", "content": GPT_USER_HINT_INTRO + f'"{user_hint}"'}
        )
    chat_messages.append({"role": "user", "content": text})

    # create API docs: https://platform.openai.com/docs/api-reference/chat/create
    response = openai.ChatCompletion.create(
        presence_penalty=-1,  # Discourage new topics
        temperature=0.2,  # Make model more predictable
        model=gpt_model,
        messages=chat_messages,
    )

    if response["choices"][0]["finish_reason"] == "content_filter":
        print(response)
        raise ValueError("ChatGPT stopped due to content filter.")

    print(response)
    estimate_input_cost = (
        estimate_total_input_tokens * GPT_PRICING[gpt_model]["input"] / 1000
    )
    input_cost = (
        response["usage"]["prompt_tokens"] * GPT_PRICING[gpt_model]["input"] / 1000
    )
    output_cost = (
        response["usage"]["completion_tokens"] * GPT_PRICING[gpt_model]["output"] / 1000
    )
    total_cost = input_cost + output_cost
    print("COSTS:")
    print(f"{input_cost=} ({estimate_input_cost=})")
    print(f"{output_cost=}")
    print(f"Total cost: ${total_cost} (~ equal to NOK {10*total_cost})")

    response_text = response["choices"][0]["message"]["content"]

    return response_text
