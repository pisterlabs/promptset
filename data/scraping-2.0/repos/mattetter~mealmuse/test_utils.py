import unittest
import time
from datetime import datetime
from mealmuse import create_app
from mealmuse.models import User, Recipe, Meal, Day, MealPlan, Pantry, ShoppingList, Item, PantryItem, ShoppingListItem, RecipeItem, db
from werkzeug.security import generate_password_hash
from mealmuse.utils import get_meal_plan
from test_data import meal_plan, recipes, shopping_list, meal_plan_biggus, add_meal_plan_to_database
import os
import openai

# MODEL = "gpt-3.5-turbo"
# openai.api_key = os.getenv("OPENAI_API_KEY")


class TestUtilsFunctions(unittest.TestCase):

    def setUp(self):
        self.app = create_app('config.DevelopmentConfig')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()
        db.drop_all()
        db.create_all()

        # Create a test user
        user = User(id=1, username="testuser", email="testuser@email.com", password=generate_password_hash("testpassword"))
        db.session.add(user)
        db.session.commit()
        db.session.close()  # Close the session after committing



    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_get_meal_plan(self):

        user = db.session.get(User, 1)
        # meal_plan = add_meal_plan_to_database(user)
        # meal_plan_id = meal_plan.id
        # user_id = user.id
        # # Check if the meal plan contains the correct number of days
        # self.assertEqual(len(meal_plan.days), 2)
        # # run get meal plan in celery
        # get_meal_plan(meal_plan_id, user_id)
        # # test to check if checking sqlalchemy object in different thread messes up the write
        # # every 5 seconds for 25 seconds or until successful, load the meal_plan_obj, check if all the recipes in it have instructions
       
        # recipe_name_list = []
        # for attempt in range(22):
            
        #     meal_plan_obj = db.session.get(MealPlan, meal_plan_id)
        #     for recipe in meal_plan_obj.recipes:
        #         if recipe.instructions:
        #             recipe_name_list.append(recipe.name)
        #     if len(recipe_name_list) == 6:
        #         print("success")
        #         print(recipe_name_list)
        #         break
        #     else: 
        #         print("not yet....")
        #         print(recipe_name_list)
        #         recipe_name_list = []
        #         time.sleep(5)
        #         db.session.close()
        













    # def test_save_meal_plan_happy_path(self):
        
    #     user = db.session.get(User, 1)
    #     # Save the meal plan to the test user
    #     save_meal_plan_to_user(meal_plan_biggus, user)

    #     # Validate that the meal plan was saved correctly
    #     self.assertIsNotNone(user.meal_plans)
    #     meal_plan = user.meal_plans[0]
    #     self.assertEqual(len(meal_plan.days), 2)

    #     for day in meal_plan.days:
    #         self.assertIn(day.name, ["Tuesday", "Wednesday"])
    #         self.assertGreater(len(day.meal), 0)
        
    #     recipe = db.session.get(Recipe, 2)
    
    #     # Check if the recipe exists
    #     if not recipe:
    #         print("Recipe not found!")
    #         return


    #     # # Print basic recipe details
    #     # print(f"Recipe Name: {recipe.name}")
    #     # print(f"Instructions: {recipe.instructions}")
    #     # print(f"Rating: {recipe.rating}")

    #     # # Print associated ingredients and their quantities
    #     # print("\nIngredients:")
    #     # for recipe_item in recipe.recipe_items:
    #     #     item = recipe_item.item
    #     #     print(f"{item.name}: {recipe_item.quantity} {recipe_item.unit}")

    # def test_add_items_to_shopping_list(self):
    #     # 1. Generate a meal plan
    #     user = db.session.get(User, 1)
    #     meal_plan = save_meal_plan_to_user(meal_plan_biggus, user)

    #     recipe = db.session.get(Recipe, 3)
    
    #     # Check if the recipe exists
    #     if not recipe:
    #         print("Recipe not found!")
    #         return

    #     recipe_id = recipe.id
                
    #     add_recipe_to_shopping_list(recipe_id, user)

    #     # Optionally, print out the shopping list for verification
    #     shopping_list = db.session.query(ShoppingList).filter_by(user_id=user.id).first()
    #     assert shopping_list is not None

    #     # for item in shopping_list.shopping_list_items:
    #     #     print(item.item.name, item.quantity, item.unit)


    # def test_add_ingredient_to_user_shopping_list(self):
    #     user = db.session.get(User, 1)

    #     # Add an ingredient to an empty shopping list
    #     ingredient = {'name': 'sugar', 'quantity': 1, 'unit': 'cup'}
    #     add_ingredient_to_user_shopping_list(user, ingredient)
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'sugar').first()
    #     self.assertIsNotNone(shopping_list_item)
    #     self.assertEqual(shopping_list_item.quantity, 1)
    #     self.assertEqual(shopping_list_item.unit, 'cup')

    #     # Add the same ingredient with a different unit
    #     ingredient = {'name': 'sugar', 'quantity': 48, 'unit': 'tsp'}
    #     add_ingredient_to_user_shopping_list(user, ingredient)
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'sugar').first()
    #     self.assertIsNotNone(shopping_list_item)
    #     self.assertEqual(shopping_list_item.quantity, 2)  # Assuming 1 cup + 48 tsp = 2 cups
    #     self.assertEqual(shopping_list_item.unit, 'cup')

    #     # Add a new ingredient
    #     ingredient = {'name': 'salt', 'quantity': 1, 'unit': 'tsp'}
    #     add_ingredient_to_user_shopping_list(user, ingredient)
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'salt').first()
    #     self.assertIsNotNone(shopping_list_item)
    #     self.assertEqual(shopping_list_item.quantity, 1)
    #     self.assertEqual(shopping_list_item.unit, 'tsp')


    # def test_remove_ingredient_from_user_shopping_list(self):
    #     user = db.session.get(User, 1)

    #     # Scenario 1: Remove an ingredient that doesn't exist
    #     ingredient = {'name': 'pepper', 'quantity': 1, 'unit': 'tsp'}
    #     remove_ingredient_from_user_shopping_list(user, ingredient)
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'pepper').first()
    #     self.assertIsNone(shopping_list_item)

    #     # Scenario 2: Remove exact quantity and unit
    #     add_ingredient_to_user_shopping_list(user, {'name': 'sugar', 'quantity': 2, 'unit': 'cup'})
    #     remove_ingredient_from_user_shopping_list(user, {'name': 'sugar', 'quantity': 2, 'unit': 'cup'})
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'sugar').first()
    #     self.assertIsNone(shopping_list_item)

    #     # Scenario 3: Remove with different unit
    #     add_ingredient_to_user_shopping_list(user, {'name': 'sugar', 'quantity': 2, 'unit': 'cup'})
    #     remove_ingredient_from_user_shopping_list(user, {'name': 'sugar', 'quantity': 96, 'unit': 'tsp'})
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'sugar').first()
    #     self.assertIsNone(shopping_list_item)

    #     # Scenario 4: Remove more than existing
    #     add_ingredient_to_user_shopping_list(user, {'name': 'salt', 'quantity': 1, 'unit': 'tsp'})
    #     remove_ingredient_from_user_shopping_list(user, {'name': 'salt', 'quantity': 2, 'unit': 'tsp'})
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'salt').first()
    #     self.assertIsNone(shopping_list_item)

    #     # Scenario 5: Use remove_entirely option
    #     add_ingredient_to_user_shopping_list(user, {'name': 'oil', 'quantity': 1, 'unit': 'cup'})
    #     remove_ingredient_from_user_shopping_list(user, {'name': 'oil', 'quantity': 0.5, 'unit': 'cup'}, remove_entirely=True)
    #     shopping_list_item = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(Item.name == 'oil').first()
    #     self.assertIsNone(shopping_list_item)

    # def test_add_ingredient_to_user_pantry(self):
    #     user = db.session.get(User, 1)

    #     # Scenario 1: Add an ingredient to an empty pantry
    #     ingredient = {'name': 'flour', 'quantity': 1, 'unit': 'cup'}
    #     add_ingredient_to_user_pantry(user, ingredient)
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'flour').first()
    #     self.assertIsNotNone(pantry_item)
    #     self.assertEqual(pantry_item.quantity, 1)
    #     self.assertEqual(pantry_item.unit, 'cup')

    #     # Scenario 2: Add the same ingredient with a different unit
    #     ingredient = {'name': 'flour', 'quantity': 16, 'unit': 'tbsp'}
    #     add_ingredient_to_user_pantry(user, ingredient)
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'flour').first()
    #     self.assertIsNotNone(pantry_item)
    #     self.assertEqual(pantry_item.quantity, 2)  # Assuming 1 cup + 16 tbsp = 2 cups
    #     self.assertEqual(pantry_item.unit, 'cup')

    #     # Scenario 3: Add a new ingredient
    #     ingredient = {'name': 'rice', 'quantity': 1, 'unit': 'cup'}
    #     add_ingredient_to_user_pantry(user, ingredient)
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'rice').first()
    #     self.assertIsNotNone(pantry_item)
    #     self.assertEqual(pantry_item.quantity, 1)
    #     self.assertEqual(pantry_item.unit, 'cup')

    # def test_remove_ingredient_from_user_pantry(self):
    #     user = db.session.get(User, 1)

    #     # Scenario 1: Remove an ingredient that doesn't exist
    #     ingredient = {'name': 'chocolate', 'quantity': 1, 'unit': 'bar'}
    #     remove_ingredient_from_user_pantry(user, ingredient)
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'chocolate').first()
    #     self.assertIsNone(pantry_item)

    #     # Scenario 2: Remove exact quantity and unit
    #     add_ingredient_to_user_pantry(user, {'name': 'flour', 'quantity': 2, 'unit': 'cup'})
    #     remove_ingredient_from_user_pantry(user, {'name': 'flour', 'quantity': 2, 'unit': 'cup'})
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'flour').first()
    #     self.assertIsNone(pantry_item)

    #     # Scenario 3: Remove with different unit
    #     add_ingredient_to_user_pantry(user, {'name': 'flour', 'quantity': 2, 'unit': 'cup'})
    #     remove_ingredient_from_user_pantry(user, {'name': 'flour', 'quantity': 32, 'unit': 'tbsp'})
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'flour').first()
    #     self.assertIsNone(pantry_item)

    #     # Scenario 4: Remove more than existing
    #     add_ingredient_to_user_pantry(user, {'name': 'rice', 'quantity': 1, 'unit': 'cup'})
    #     remove_ingredient_from_user_pantry(user, {'name': 'rice', 'quantity': 2, 'unit': 'cup'})
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'rice').first()
    #     self.assertIsNone(pantry_item)

    #     # Scenario 5: Use remove_entirely option
    #     add_ingredient_to_user_pantry(user, {'name': 'oil', 'quantity': 1, 'unit': 'cup'})
    #     remove_ingredient_from_user_pantry(user, {'name': 'oil', 'quantity': 0.5, 'unit': 'cup'}, remove_entirely=True)
    #     pantry_item = db.session.query(PantryItem).join(Pantry).join(Item).filter(Item.name == 'oil').first()
    #     self.assertIsNone(pantry_item)

    # def test_remove_recipe_from_shopping_list(self):
    #      # 1. Generate a meal plan
    #     user = db.session.get(User, 1)
    #     meal_plan = save_meal_plan_to_user(meal_plan_biggus, user)
        
    #     #save a recipe to the shopping list
    #     recipe_id = 5      
    #     add_recipe_to_shopping_list(recipe_id, user)

    #     # Remove the recipe from the shopping list
    #     remove_recipe_from_shopping_list(recipe_id, user)

    #     # Check if the ingredients of the recipe are removed or reduced in quantity
    #     recipe_items = db.session.query(RecipeItem).filter_by(recipe_id=recipe_id).all()
    #     for recipe_item in recipe_items:
    #         item_in_list = db.session.query(ShoppingListItem).join(ShoppingList).join(Item).filter(
    #             ShoppingList.id == user.shopping_list.id, 
    #             Item.name == recipe_item.item.name
    #         ).first()

    #         # # If the assertion is likely to fail, print the quantities
    #         # if item_in_list and item_in_list.quantity >= recipe_item.quantity:
    #         #     print(f"Item: {recipe_item.item.name}")
    #         #     print(f"Quantity in shopping list: {item_in_list.quantity}")
    #         #     print(f"Quantity in recipe: {recipe_item.quantity}")
    #         # Assert that either the item is not in the list, or its quantity is reduced
    #         self.assertTrue(item_in_list is None or item_in_list.quantity < recipe_item.quantity)

    # def test_get_meal_plan_details(self):
    #     user = db.session.get(User, 1)

    #     # Save a sample meal plan to the test user
    #     save_meal_plan_to_user(meal_plan_biggus, user)

    #     # Get the most recent meal plan for the user
    #     meal_plan_details = get_meal_plan_details(user)

    #     # Validate that the meal plan details dictionary has the expected structure
    #     self.assertIn("meal_plan_id", meal_plan_details)
    #     self.assertIn("days", meal_plan_details)

    #     # Ensure there are two days in the meal plan
    #     self.assertEqual(len(meal_plan_details["days"]), 2)

    #     for day_data in meal_plan_details["days"]:
    #         self.assertIn("day_id", day_data)
    #         self.assertIn("day_name", day_data)
    #         self.assertIn("meals", day_data)

    #         # Ensure each day has at least one meal
    #         self.assertGreater(len(day_data["meals"]), 0)

    #         for meal_data in day_data["meals"]:
    #             self.assertIn("meal_id", meal_data)
    #             self.assertIn("meal_name", meal_data)
    #             self.assertIn("recipes", meal_data)

    #             # Ensure each meal has at least one recipe
    #             self.assertGreater(len(meal_data["recipes"]), 0)

    #             for recipe_data in meal_data["recipes"]:
    #                 self.assertIn("recipe_id", recipe_data)
    #                 self.assertIn("recipe_name", recipe_data)


    # You can add more tests for other functions if needed!
# Run the tests with: python -m unittest test_utils.py
if __name__ == '__main__':
    unittest.main()