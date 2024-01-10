from openai import OpenAI
import argparse


def main():
    print("Recipehelper is running!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    
    # print("List of products from input:", user_input)
    generate_recipe(user_input)


def generate_recipe(prompt: str):
    client = OpenAI()

    subject = prompt

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a generetor of recipe for given products"},
        {"role": "user", "content": f'I have only these products: {subject}. Suggest me a recipe only for these products, I do not want to go to the store.'},
    ]
    )
    AI_answer = completion.choices[0].message
    recipe = AI_answer.content
    # print(recipe, 'recipe')
    return recipe
    


if __name__ == "__main__":
    main()
