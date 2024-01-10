import logging
import json

from fastapi import APIRouter, Form, HTTPException, Header
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.chains import LLMChain
from square.client import Client

from ..settings.config import Config
from ..utils.clean import cleaned
from typing import Optional, Annotated, Union


# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config.get_instance()
conn = config.get_postgres_connection()
llm = config.get_vertex_ai_connection()
dalle_llm = config.get_open_ai_connection()
chatbot_llm = config.get_openai_text_connection()

router = APIRouter()


@router.post("/recommend_menu", tags=["seller"])
def recommend_menu(preferred_cuisine: str = Form(...), prep_time_breakfast: str = Form(...),
                   prep_time_lunch: str = Form(...), prep_time_dinner: str = Form(...),
                   cook_time_breakfast: str = Form(...), cook_time_lunch: str = Form(...),
                   cook_time_dinner: str = Form(...)):
    """
    Recommend menu using Vertex AI
    :param cook_time_dinner:
    :param preferred_cuisine:
    :param prep_time_breakfast:
    :param prep_time_lunch:
    :param prep_time_dinner:
    :param cook_time_breakfast:
    :param cook_time_lunch:
    :param cook_time_dinner:
    :return: Detailed Menu JSON
    """
    logger.info(f"Reading ingredients from Ingredient table of postgres")
    try:
        cur = conn.cursor()
        cur.execute("""SELECT * FROM "Ingredients" """)
        rows = cur.fetchall()
        # Column Names - id, Created_at,ingredient_name,ingredient_type, ingredient_sub_type, shelf_life_days, quantity, unit
        logger.info(f"Total number of ingredients in the table: {len(rows)}")
        # Create a list of Dictionary of Ingredients with Name, quantity unit, shelf life days, ingredient type,
        # ingredient sub type
        ingredients = []
        for row in rows:
            ingredient = {'name': row[2], 'quantity': row[6], 'unit': row[8], 'shelf_life_days': row[5],
                          'ingredient_type': row[3], 'ingredient_sub_type': row[4], 'ingredient_id': row[0],
                          'unitprice': row[7]}
            ingredients.append(ingredient)
        logger.info(f"Read ingredients from Postgres")

    except Exception as e:
        # Raise HTTP exception
        logger.exception(f"An Exception Occurred while reading ingredients from Postgres --> {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Pass the list to Context and generate the menu
    template = """ 
     **Context:**
You are a chef at a {preferred_cuisine} restaurant. Your kitchen is stocked with essential ingredients such as flour, water, spices, milk, curd, onion, tomato, ginger, garlic, oil, butter, and ghee, each with its unit price. Your mission is to craft a menu for the restaurant that reflects your culinary style. 

**Task:**
Prepare a comprehensive menu featuring at least 25 dishes per categories: Breakfast, Lunch, Dinner, Dessert, Drinks, Sides, and Breads. For each category, there must be a minimum of 10 dishes. Additionally, ensure that the menu itemizes the price of each dish.

**Price Calculation:**
- The cost of one ingredient is determined using the formula: Price = (Quantity * Unit Price). For instance, if a dish requires 2 potatoes, and each potato costs 10, the price of potatoes for that dish is 20. This calculation applies to all ingredients.

- The price of a dish is calculated using the formula: Price = (Sum of Price of all ingredients + 30% of the sum + 5% of the sum) + 10% tax. For example, if the total cost of all ingredients needed for a dish is 100, the price of the dish is 135 (100 + 30 + 5) + 10% tax, making it 148.5 in total.

**Answer:**
Provide the menu in JSON key-value (Keys are Course, Dish name, Customization & Price) pairs without special characters. This includes the course name, dish name, its price, and any customizations based on the ingredients. Ensure that there are at least 3 dishes for each category. Here's the output format:

```
(Breakfast: 
  (Dish1: 
    (Customization: ["Option1", "Option2"], 
    Price: amount), 
  Dish2: 
    (Price: amount)), 
(Lunch: 
  (Dish3: 
    (Price: amount, 
    Customization: ["No onion"]), 
  Dish4: 
    (Price: amount)), 
(Dinner: 
  (Dish5: 
    (Price: amount, 
    Customization: ["Low spice", "No onion"]), 
  Dish6: 
    (Price: amount)), 
(Dessert: 
  (Dish7: 
    (Price: amount), 
  Dish8: 
    (Price: amount)), 
(Drinks: 
  (Dish9: 
    (Price: amount), 
  Dish10: 
    (Price: amount)), 
(Sides: 
  (Dish11: 
    (Price: amount), 
  Dish12: 
    (Price: amount)), 
(Breads: 
  (Dish13: 
    (Price: amount), 
  Dish14: 
    (Price: amount)))
```

**Example:**
```
(
  "Breakfast": (
    "Scrambled Eggs": (
      "Customization": ["Cheese", "Bacon"],
      "Price": 8.5
    ),
    "Pancakes": (
      "Price": 7.0
    ),
    "Omelette": (
      "Customization": ["Spinach", "Tomato"],
      "Price": 9.0
    )
  ),
  "Lunch": (
    "Chicken Curry": (
      "Price": 12.0,
      "Customization": ["Spicy", "No Onion"]
    ),
    "Vegetable Biryani": (
      "Price": 10.5
    ),
    "Grilled Cheese Sandwich": (
      "Customization": ["Extra Cheese"],
      "Price": 6.5
    )
  ),
  "Dinner": (
    "Steak": (
      "Price": 18.0,
      "Customization": ["Medium Rare", "No Onion"]
    ),
    "Salmon Fillet": (
      "Price": 15.5
    ),
    "Pasta Carbonara": (
      "Customization": ["Extra Bacon"],
      "Price": 13.0
    )
  ),
  "Dessert": (
    "Chocolate Cake": (
      "Price": 6.5
    ),
    "Tiramisu": (
      "Price": 8.0
    ),
    "Fruit Salad": (
      "Price": 5.0
    )
  ),
  "Drinks": (
    "Margarita": (
      "Price": 9.0
    ),
    "Mojito": (
      "Price": 7.5
    ),
    "Iced Tea": (
      "Price": 3.5
    )
  ),
  "Sides": (
    "Garlic Bread": (
      "Price": 4.0
    ),
    "French Fries": (
      "Price": 4.0
    ),
    "Coleslaw": (
      "Price": 3.0
    )
  ),
  "Breads": (
    "Garlic Naan": (
      "Price": 2.5
    ),
    "Whole Wheat Roti": (
      "Price": 2.0
    ),
    "Ciabatta": (
      "Price": 2.0
    )
  )
)

```

**Constraints:**
- Ensure that the menu aligns with the {preferred_cuisine}.
- Breakfast, lunch, and dinner preparation times must not exceed {prep_time_breakfast}, {prep_time_lunch}, and {prep_time_dinner} respectively.
- Breakfast, lunch, and dinner cooking times should not surpass {cook_time_breakfast}, {cook_time_lunch}, and {cook_time_dinner} respectively.
- Just output Course, Dish name, Customization & Price in the JSON key-value pairs. Do not include the recipe or ingredients.

**Definitions:**
- Prep time: The time taken to prepare the dish.
- Cook time: The time taken to cook the dish. 

Just output Course, Dish name, Customization & Price in the JSON key-value pairs. Do not include the recipe or ingredients or code or any other text."""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | chatbot_llm

    # Generate the menu
    try:
        s =  chain.invoke({'preferred_cuisine': preferred_cuisine, 'ingredients': ingredients,
                             'prep_time_breakfast': prep_time_breakfast, 'prep_time_lunch': prep_time_lunch,
                             'prep_time_dinner': prep_time_dinner, 'cook_time_breakfast': cook_time_breakfast,
                             'cook_time_lunch': cook_time_lunch, 'cook_time_dinner': cook_time_dinner})
        try:
            return json.loads(cleaned(s))
        except Exception as e:
            print(e)
            return cleaned(s)
    except Exception as e:
        logger.exception(f"An Exception Occurred while generating menu using Vertex AI --> {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Read All ingredients from Ingredient table of postgres, and recommend menu using Vertex AI
@router.post("/add_ingedients", tags=["seller"])
def add_ingedients(ingredient_name: str = Form(...), ingredient_type: str = Form(...),
                   ingredient_sub_type: str = Form(...), shelf_life_days: str = Form(...), quantity: str = Form(...),
                   unit: str = Form(...), unitprice: Optional[str]=Form(None) ):
    """
    Add ingredients to Ingredient table of postgres
    :param ingredient_name:
    :param ingredient_type:
    :param ingredient_sub_type:
    :param shelf_life_days:
    :param quantity:
    :param unit:
    :param unitprice:
    :return: Success message
    """
    try:
        #dataprocessing
        if unitprice is None:
            unitprice = 2
        ingredient_name = ingredient_name.lower()
        ingredient_type = ingredient_type.lower()
        ingredient_sub_type = ingredient_sub_type.lower()
        unit = unit.lower()


        cur = conn.cursor()
        cur.execute(
            """INSERT INTO "Ingredients"(ingredient_name,ingredient_type,ingredient_sub_type,shelf_life_days,quantity,units,unitprice) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (ingredient_name, ingredient_type, ingredient_sub_type, shelf_life_days, quantity, unit, unitprice))
        conn.commit()
        logger.info(f"Added ingredients to Postgres")
        return {"message": "Ingredient added successfully"}
    except Exception as e:
        # Raise HTTP exception
        logger.exception(f"An Exception Occurred while adding ingredients to Postgres --> {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_ingredient_summary", tags=["seller"])
def get_ingredient_summary():
    """
    Get ingredient summary from Ingredient table of postgres
    :return: ingredient summary
    """
    try:
        cur = conn.cursor()
        cur.execute("""SELECT * FROM "Ingredients" """)
        rows = cur.fetchall()
        # Column Names - id, Created_at,ingredient_name,ingredient_type, ingredient_sub_type, shelf_life_days, quantity, unit
        logger.info(f"Total number of ingredients in the table: {len(rows)}")
        ingredients = []
        for row in rows:
            ingredient = {'name': row[2], 'quantity': row[6], 'unit': row[8], 'shelf_life_days': row[5],
                          'ingredient_type': row[3], 'ingredient_sub_type': row[4], 'ingredient_id': row[0],
                          'unitprice': row[7]}
            ingredients.append(ingredient)
        logger.info(f"Read ingredients from Postgres, Summarizing")

        # Make summary with Vertex AI
        template = """ 
        CONTEXT: You are an AI bot provided with a list of ingredients {ingredients}. You need to sum up , group & summarize the list of ingredients.
        TASK: Group up all Ingredients based on their name and type , sum up their quantity and provide a summary of the ingredients.
        ANSWER: Provide the JSON key-value pairs without special chars 
                (
            "ingredient_type1": (
                "name": "ingredient_name1",
                "quantity": "quantity"
            ),
            "ingredient_type2": (
                "name": "ingredient_name2",
                "quantity": "quantity"
            )
        )

        For example: 
                (
            "Vegetables": (
                (name: "Tomato", quantity: 10),
                (name: "Potato", quantity: 20)
            ),
            "Spices": (
                (name: "Salt", quantity: 10),
                (name: "Pepper", quantity: 20)
            )
        )

        CONSTRAINTS: Keep in mind the summary should be based on ingredient name and type.
        In case of similar names like Tomato and Tomato Puree, group them together, Add their Quantities, like Potato 500 gram, Potato 500 g, Potato 1kg,Potato  8 kilogram, group all and  sum them up like Potato: 10kg.
        Provide JSON only, no recipie, no code, no special chars, no extra text.
        """

        prompt = PromptTemplate.from_template(template)
        chain = prompt | chatbot_llm

        # Generate the summary
        try:
            s =  chain.invoke({'ingredients': json.dumps(ingredients)})
            try:
                return json.loads(cleaned(s))
            except Exception as e:
                print(e)
                return cleaned(s)


        except Exception as e:
            logger.exception(f"An Exception Occurred while generating summary using Vertex AI --> {e}")
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        # Raise HTTP exception
        logger.exception(f"An Exception Occurred while reading ingredients from Postgres --> {e}")
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/reengineer_dish", tags=["seller"])
def reengineer_dish(dish_name: str = Form(...), preferred_cuisine: str = Form(...)):
    """
    Reengineer the dish with same ingredients
    :param dish_name:
    :return: New dish name
    """
    template = """ Context: You are a chef of a {preferred_cuisine} restaurant. You are given a dish that you need to reengineer. Please recommend some other dish, that has same ingredients as {dish_name}. You always have flour, water, spices, milk, curd, onion, tomato, ginger, garlic, oil, butter, ghee in your inventory,
        Task: Reengineer the dish with same ingredients. 
        Answer: Provide a single dish name and price only in JSON, no recipie.
        EXAMPLE: ("Dish": "Shahi Panner", "Price":30) in JSON
        Constraints: Keep in mind the dish should be {preferred_cuisine} dish and prepration and cook time of new and old dish should be similar.
        Definations: Prep time is the time taken to prepare the dish. Cook time is the time taken to cook the dish."""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    try:
        s = chain.invoke({'dish_name': dish_name, 'preferred_cuisine': preferred_cuisine})
        try:
            return json.loads(cleaned(s))
        except Exception as e:
            print(e)
            return cleaned(s)
    except Exception as e:
        logger.exception(f"An Exception Occurred while reengineering dish using Vertex AI --> {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/catalog_image_generator")
def catalog_image_generator(dish_name: str = Form(...), image_type: Optional[str] = Form(None)):
    """
    Generate image for the dish
    :param dish_name:
    :param image_type:
    :return:
    """
    image_type = image_type or "realistic"
    template = """ Context: You are an AI bot responsible for Image generation of dishes in a Restaraunt. 
    TASK: You are given a dish {dish_name}. Please generate a Prompt to generate {image_type}, well-plated, mouth-watering and tempting image for the dish to display the serving suggestions.
    Answer: Provide the Prompt 
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(prompt=prompt, llm=dalle_llm)
    try:
        image_url = DallEAPIWrapper().run(chain.run({'dish_name': dish_name, 'image_type': image_type}))
        logger.info(f"Image generated successfully")
        return {"image_url": image_url}
    except Exception as e:
        logger.exception(f"An Exception Occurred while generating image using Open AI --> {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_seller_location", tags=["seller"])
def get_seller_info(access_token: Annotated[Union[str, None], Header()]):
    info = Client(access_token=access_token, environment='sandbox')
    result = info.locations.list_locations()
    print(result)
    if result.is_success():
        square_location_id = result.body['locations'][0]['id']
        logger.info(f"Connected to Square")
        return {"location_id":square_location_id}
    elif result.is_error():
        for error in result.errors:
            raise Exception(
                f"Error connecting to Square --> Category :{error['category']} Code: {error['code']} Detail: {error['detail']}")







