from typing import List

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from config.open_ai import config as openai_config
from classes import IngredientList
from utils.read_file import get_ingredients as get_ingredients_data
from schemas.ingredients import IngredientsResponse

import openai

router = APIRouter()

@router.post("/recipe")
def get_recipe(ingredientsList: List[IngredientList] = Body(..., embed=True)):
    try:
        
        all_ingredients = []
        for ingredientList in ingredientsList:
            for ingredient in ingredientList['ingredients']:
                all_ingredients.append(ingredient)

        complete_prompt = (
            f"{openai_config['prompt']} {', '.join(all_ingredients)}. "
            "INTENTA USAR TODOS LOS INGREDIENTES. \n"
            "Recuerda listar los ingredientes y las instrucciones. \n"
            "Y si no pudiste ocupar ingredientes, mencionalo indicando cuales, pero solo eso, no otro detalle, si ocupaste todos los ingredientes entonces no lo menciones. \n"
            "Dame SIEMPRE la respuesta con tags html como <h1>, <h2>, <h3>, <p>, <li>, <il>, etc y <br>, y no hagas mención de esto en la respuesta."
            "El titulo de la receta en <h1>, el titulo de los ingredientes como <h3>, los ingredientes como una lista con <li>, las instrucciones como una lista con <li>."
            "Cualquier otro texto como <p>. Puedes usar <b>. \n"
        )

        completion = openai.ChatCompletion.create(
            model=openai_config['model'],
            messages=[{"role": "user", "content": complete_prompt}]
        )

        response_data = {"data": completion.choices[0].message.content}

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
