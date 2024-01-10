import os

import openai
from dotenv import load_dotenv

RECIPE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ingredient": {
                        "description": "材料",
                        "type": "string",
                        "examples": ["鶏もも肉"],
                    },
                    "quantity": {
                        "description": "分量",
                        "type": "string",
                        "examples": ["300g"],
                    },
                },
                "required": ["ingredient", "quantity"],
            },
        },
        "steps": {
            "type": "array",
            "description": "手順",
            "items": {"type": "string"},
            "examples": [["材料を切ります。", "材料を炒めます。"]],
        },
    },
    "required": ["ingredients", "steps"],
}

OUTPUT_RECIPE_FUNCTION = {
    "name": "output_recipe",
    "description": "レシピを出力する",
    "parameters": RECIPE_JSON_SCHEMA,
}


def main():
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "カレーのレシピを教えてください。"}],
        functions=[OUTPUT_RECIPE_FUNCTION],
        function_call={"name": OUTPUT_RECIPE_FUNCTION["name"]},
    )

    response_message = response["choices"][0]["message"]
    function_call_name = response_message["function_call"]["name"]
    function_call_args = response_message["function_call"]["arguments"]

    print(function_call_name)
    print(function_call_args)


if __name__ == "__main__":
    main()
