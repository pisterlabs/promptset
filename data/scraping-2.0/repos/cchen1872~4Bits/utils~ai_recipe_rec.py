import openai
import json

# Initialize the OpenAI API with your API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def get_recipe_recommendation(requirements):
    """
    Get a recipe recommendation based on user requirements using GPT-3.5.

    Parameters:
    - requirements (str): The user's requirements for the recipe.

    Returns:
    - str: A recommended recipe based on the user's requirements.
    """
    # Define the prompt for GPT-3.5
    prompt = f"Based on the following requirements, please recommend a recipe:\n\n{requirements}\n\nRecipe:"

    # Make the API call
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=200  # You can adjust this value based on your needs
    )

    # Extract the recommended recipe from the response
    recommended_recipe = response.choices[0].text.strip()

    return recommended_recipe

if __name__ == "__main__":
    # Get user requirements
    user_requirements = input("Please enter your recipe requirements: ")

    # Get the recommended recipe
    recipe = get_recipe_recommendation(user_requirements)

    # Print the recommended recipe
    print("\nRecommended Recipe:")
    print(recipe)

