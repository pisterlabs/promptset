import semantic_kernel as sk
kernel = sk.Kernel()

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

cooking_skill = kernel.import_semantic_skill_from_directory("plugins", "Cooking")
recipe_function = cooking_skill["RecipeGenerator"]

# print(recipe_function("Chicken Adobo Filipino Style"))

marketing_skill = kernel.import_semantic_skill_from_directory("plugins", "Marketing")
advert_function = marketing_skill["AdvertisementGenerator"]

recipeResult = recipe_function("Chicken Adobo Filipino Style").result

advertResult = advert_function(recipeResult).result

print("Recipe: " + recipeResult)
print("Advert: " + advertResult)