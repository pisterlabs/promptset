import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion



def getAIRecipe(mealname, ingredient_list):
    kernel = sk.Kernel()
# Prepare OpenAI service using credentials stored in the `.env` file
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))
    # Wrap your prompt in a function
    prompt = kernel.create_semantic_function(f"""
    I need you to generate a recipe/tutorial/instruction for a creative recipe of a dish called {mealname}, and the ingredient list includes {ingredient_list}. Have steps in bulletpoints, each step is written starting with a asteriskmark "*" in the beginning, except for the first step. Don't need to list ingredients, just instruction. Don't include order number of the steps, just start each step with a questionmark. Keep it short and concise, around 6-8 steps, and less than 150 words. Format of a sample recipe should be like this: Preheat oven to 375 degrees. *In a medium bowl, toss bread with butter until coated. *In a large bowl, toss together apples, brown sugar, maple syrup, nutmeg, prunes, brandy, juice, and half the bread until combined. *Transfer mixture to an 8-inch square baking dish and top with remaining bread. *Cover with foil and bake until mixture is bubbling and apples are tender, about 40 minutes. *Uncover and bake until top is browned, 10 to 15 minutes. *Let cool 15 minutes. Serve with ice cream or whipped cream.
    """)

    result = prompt().result
    print(result)
    return result


