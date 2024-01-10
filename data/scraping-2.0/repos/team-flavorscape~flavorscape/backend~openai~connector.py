import sys
from openai import OpenAI

TAGS = ['Contains shellfish', 'hf-originals', 'Spicy', 'sides-bites', 'Easy Prep', 'sides-bread', 'labor-day-sale', 'blackfriday', 'One Pot', 'destination-italy', 'Climate Superstar', 'sides-soup', 'chicken-wing', 'lunch-quiche', 'Child friendly', 'pumpkin-spices', 'lunch-sale', 'dinners', 'Protein smart', 'apple-lovers', 'family-dinner-faves', 'Under 30 Minutes', 'fall-flavours', 'Extra spicy', 'Organic', 'Carb smart', '4 Servings', 'dessert', 'Summer', 'Multi-Portion', 'pasta-night', 'birthday-party', 'breakfast-essentials', 'Gluten-free', 'breakfast-kits', 'Lunch + Snacks', 'lunch-salad', 'Bestseller', 'grocery-bakery', 'Plant-Based Protein', 'lunches', 'sides', '3-Course Meal', 'free-addon', 'Dinner Ideas', 'Vegan', 'sides-boards', 'Quick', 'Good Climate Score', 'Fair Climate Score', 'game-day', 'Dietitian Win', 'summer-alfresco', 'breakfast-bakery', 'Veggie', 'waffle-day', 'weeknight-meals', 'Family Friendly', 'sides-proteins', 'grocery', 'breakfast-promo', 'sides-salad', 'grocery-snacks', 'specialties', 'lunch-pasta', 'kids-corner', 'Calorie Smart', 'Rapid', 'New', 'breakfasts', 'staff-picks', 'lunch-pizza', 'lunch-soup', 'Lightning Prep', 'Easy', 'Meal Prep', 'dessert-ready', 'Carb Smart', 'date-night', "Chef's Choice", 'Eat First', 'Oven Ready', 'Protein Smart', 'One Pan', 'halloween', 'Mediterranean', 'lunch-sandwiches', 'sides-veggie', 'lunch-combos', 'Sheet Pan', 'grocery-proteins', 'No Oven', 'latin-heritage', 'Seasonal', 'Cook together', 'Easy Cleanup', 'beach-day']

class ChatGPTConnector():
    def __init__(self):
        self.client = OpenAI()

    def get_tags_for_prompt(self, prompt, tags):
        completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your are part of a recommendation system for recipes. Your task is to select tags describing recipes matching the user's prompt from the following comma-separated list: " + ','.join(TAGS) + ". Respond in exactly the following format: \"tag1, tag2, tag3\". You are allowed to respond with zero to five tags."},
            {"role": "user", "content": prompt}
          ]
        )

        print(completion.choices[0].message)

if __name__ == '__main__':
    prompt = sys.argv[1]
    c = ChatGPTConnector()
    c.get_tags_for_prompt(prompt, TAGS)
