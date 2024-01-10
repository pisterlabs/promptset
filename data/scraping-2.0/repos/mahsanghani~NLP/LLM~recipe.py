import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a recipe based on these ingredients and instructions:\n\nFrito Pie\n\nIngredients:\nFritos\nChili\nShredded cheddar cheese\nSweet white or red onions, diced small\nSour cream\n\nInstructions:",
  temperature=0.3,
  max_tokens=120,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# biryani recipe

# Sample response

# Sure! Biryani is a flavorful and aromatic rice dish that originated in the Indian subcontinent. There are many variations of biryani, but I'll provide you with a basic recipe for chicken biryani. Here's how you can make it:

# Ingredients:

# 2 cups basmati rice
# 500 grams chicken, cut into pieces
# 2 onions, thinly sliced
# 2 tomatoes, chopped
# 4 cloves of garlic, minced
# 1-inch piece of ginger, grated
# 2 green chilies, slit
# 1/2 cup plain yogurt
# 1/2 cup chopped fresh cilantro (coriander leaves)
# 1/2 cup chopped fresh mint leaves
# 1 teaspoon red chili powder
# 1/2 teaspoon turmeric powder
# 1 teaspoon biryani masala powder
# 1/2 teaspoon garam masala powder
# 4 cups water
# 4 tablespoons ghee (clarified butter) or cooking oil
# Salt to taste
# Saffron strands (optional)
# Warm milk (optional)
# For the marinade:

# 1/2 cup plain yogurt
# 1 tablespoon ginger-garlic paste
# 1 teaspoon red chili powder
# 1/2 teaspoon turmeric powder
# Juice of 1/2 lemon
# Salt to taste
# Instructions:

# Rinse the basmati rice under cold water until the water runs clear. Then, soak the rice in water for 30 minutes. After 30 minutes, drain the rice and set it aside.

# In a large bowl, combine the chicken pieces with all the marinade ingredients listed above. Mix well to coat the chicken evenly. Let it marinate for at least 30 minutes or refrigerate for a few hours for better flavor.

# Heat 2 tablespoons of ghee or oil in a large, heavy-bottomed pot over medium heat. Add the sliced onions and cook until they turn golden brown and crispy. Remove half of the fried onions from the pot and set them aside for garnishing.

# To the remaining onions in the pot, add the minced garlic, grated ginger, and slit green chilies. Sauté for a minute until fragrant.

# Add the chopped tomatoes and cook until they become soft and mushy.

# Now, add the marinated chicken to the pot and cook for about 5-6 minutes until the chicken changes color and is partially cooked.

# In a separate small bowl, whisk together the yogurt, red chili powder, turmeric powder, biryani masala powder, and salt. Add this mixture to the chicken and stir well to combine. Cook for another 5 minutes, allowing the flavors to blend.

# Add the chopped cilantro (coriander leaves) and mint leaves to the pot and mix well.

# In a separate large pot, bring 4 cups of water to a boil. Add the soaked and drained rice to the boiling water. Cook the rice until it is 70-80% done (slightly undercooked). This step is crucial as the rice will continue cooking later with the chicken.

# Drain the partially cooked rice and layer it over the chicken in the pot. Ensure that the rice is spread evenly.

# Drizzle the remaining 2 tablespoons of ghee or oil over the rice. If desired, you can dissolve a few saffron strands in warm milk and drizzle it over the rice for added flavor and color.

# Cover the pot tightly with a lid and cook on low heat for about 20-25 minutes. This will allow the flavors to meld, and the rice will fully cook and become