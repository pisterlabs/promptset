import os
import openai

# 環境変数からAPIキーを取得
openai_api_key = os.environ.get("OPENAI_API_KEY")

# APIキーが存在しない場合、エラーメッセージを表示して終了
if openai_api_key is None:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    exit(1)

# OpenAI APIにAPIキーを設定
openai.api_key = openai_api_key

def request_recipe(ingredients_list):
    ingredients_text = ', '.join(ingredients_list)
    prompt = f"3 dishes with {ingredients_text}?"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    recipes_text = response.choices[0].text.strip()
    recipes = recipes_text.split(", ")
    return recipes


def request_instructions(recipe_name):
    prompt = f"Please provide a brief recipe for {recipe_name}, including only the main ingredients and essential steps."
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    instructions = response.choices[0].text.strip()

    return instructions
