import os
import json
import openai
from controller.object_detection_controller import object_detection_controller
from helpers.dalle_helper import DallEHelper

openai.api_key = os.getenv("OPENAI_API_KEY")


class RecipeController():
    def __init__(self) -> None:
        pass

    def create_recipe_from_prompt_text(self, prompt_text):
        prompt_start = "Sana verdiğim malzeme listesi ve bilgiler dahilinde bana bir yemek tarifi öner.\n"
        prompt = prompt_start + prompt_text
        prompt += """\nTarifi bu formatta olustur. : {"name": "yemek ismi", "ingredients": ["1 bardak sut", "1 çorba kaşığı un"], "instructions" : " ornek ornek"} \n"""

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                    "content":
                    """Bir aşçı gibi davran. Sana verdiğim malzeme listesinden bana bir yemek tarifi oluştur. Verdiğim listeye sadık kalmaya çalış. Malzemelerin hepsini kullanmak zorunda değilsin. Her evde bulunabilecek malzemeleri de var kabul edebilirsin.Bir aşçı gibi davran. Sana verdiğim malzeme listesinden bana bir yemek tarifi oluştur. Verdiğim listeye sadık kalmaya çalış. Malzemelerin hepsini kullanmak zorunda değilsin. Her evde bulunabilecek malzemeleri de var kabul edebilirsin. Yemeğin adı, içeriği ve yapılışını aşşağıdaki JSON formatinda ver bunun dışında bir şey yazma"""},
                    {"role": "user", "content": prompt}, ])
        print(response["choices"][0]["message"]["content"])
        response_text = json.loads(response["choices"][0]["message"]["content"])
        dh = DallEHelper(os.getenv("OPENAI_API_KEY"))
        image_path = dh.create_image(response_text["name"])
        print({"recipe": response_text, "image": image_path.split('/')[1]})
        return {"recipe": response_text, "image": image_path.split('/')[1]}


    def create_recipe_from_image(self, image_path, choices):
        print(image_path, choices)
        choice_dict = {
            "isVeganSelected": "vegan",
            "isVegetarianSelected": "vejetaryen",
            "isGlutenFreeSelected": "glutensiz",
            "isKetoSelected": "keto diyete uygun",
            "isLowCarbSelected": "düşük karbonhidratlı",
            "isLowFatSelected": "düşük yağlı",

        }
        ingredients = object_detection_controller.get_fridge_contents(image_path)
        prompt = f"Sana verdiğim malzeme listesinden bana bir {choices['recipeType']} tarifi öner Bu tarif {', '.join([choice_dict[item] for item in choice_dict.keys() if choices[item]])} bir tarif olsun:\n"
        print(prompt)
        for ingredient in ingredients["ingredients"]:
            prompt += f"- {ingredient}\n"
        prompt += """\nTarifi bu formatta olustur. : {"name": "yemek ismi", "ingredients": ["1 bardak sut", "1 çorba kaşığı un"], "instructions" : " ornek ornek"} \n"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content":
                       """Bir aşçı gibi davran. Sana verdiğim malzeme listesinden bana bir yemek tarifi oluştur. Verdiğim listeye sadık kalmaya çalış. Malzemelerin hepsini kullanmak zorunda değilsin. Her evde bulunabilecek malzemeleri de var kabul edebilirsin.Bir aşçı gibi davran. Sana verdiğim malzeme listesinden bana bir yemek tarifi oluştur. Verdiğim listeye sadık kalmaya çalış. Malzemelerin hepsini kullanmak zorunda değilsin. Her evde bulunabilecek malzemeleri de var kabul edebilirsin. Yemeğin adı, içeriği ve yapılışını aşşağıdaki JSON formatinda ver bunun dışında bir şey yazma"""},
                      {"role": "user", "content": prompt}, ])
        print(response["choices"][0]["message"]["content"])
        response_text = json.loads(response["choices"][0]["message"]["content"])
        dh = DallEHelper(os.getenv("OPENAI_API_KEY"))
        image_path = dh.create_image(response_text["name"])
        print({"recipe": response_text, "image": image_path.split('/')[1], "detected_objects": ingredients["ingredients"]})
        return {"recipe": response_text, "image": image_path.split('/')[1], "detected_objects": ingredients["ingredients"]}


recipe_controller = RecipeController()
