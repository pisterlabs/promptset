# # Parser using Custom Output Parser

# from langchain import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# import os
# import json
# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# # The data source
# food_label = '/Users/arshad/Desktop/Projects/Healthy-Eats/sample_image/sample.jpeg'


# # Prompt Template
# ingredients_template = PromptTemplate(
#     input_variables=['food_label'],
#     template="""You are a great Ingredient Parser who can extract ingredients from a given food label text.
#     Extract the ingredients from the following food_label:
#     FOOD LABEL: {food_label}"""
# )

# template_string = """You are a master ingredient parser from a given food label. You give detailed descriptions of the ingredients\
# You can classify each ingredient as Healthy/Unhealthy.
# You also add emojis for each ingredient.

# Take the Food Label below delimited by triple backticks and use it to extract the ingredients and provide a detailed description.

# brand description: ```{food_label}```

# then based on the description you give the brand an Emoji and a label for healthy or unhelathy.

# Format the output as JSON with the following keys:
# Ingredient
# Description
# Emoji
# Healthy/Unhealthy label
# """

# prompt_template = ChatPromptTemplate.from_template(template_string)

# chat_llm = ChatOpenAI(temperature=0.0)
# llm = OpenAI(temperature=0)
# ingredients_chain = LLMChain(
#     llm=llm, prompt=ingredients_template, verbose=True, output_key='ingredients')

# ingredients_list = prompt_template.format_messages(
#     food_label=ingredients_chain.run(food_label))

# response = chat_llm(ingredients_list)

# final_response = response.content

# data_dict = json.loads(final_response)


x = {'ingredients': [{'ingredient': 'Rice Flour', 'description': 'Rice flour is a fine powder made from ground rice. It is commonly used as a gluten-free alternative to wheat flour.', 'emoji': '🍚', 'label': 'Healthy'}, {'ingredient': 'Corn Flour', 'description': 'Corn flour is a fine powder made from ground corn kernels. It is commonly used as a thickening agent in cooking and baking.', 'emoji': '🌽', 'label': 'Healthy'}, {'ingredient': 'Edible Vegetable Oil', 'description': 'Edible vegetable oil refers to any oil that is derived from plants and can be consumed. Common examples include olive oil, canola oil, and sunflower oil.', 'emoji': '🌿', 'label': 'Healthy'}, {'ingredient': 'Gram Flour', 'description': 'Gram flour, also known as chickpea flour or besan, is a flour made from ground chickpeas. It is commonly used in Indian and Middle Eastern cuisines.', 'emoji': '🌱', 'label': 'Healthy'}, {'ingredient': 'Salt', 'description': 'Salt is a mineral composed primarily of sodium chloride. It is used to enhance the flavor of food.', 'emoji': '🧂', 'label': 'Unhealthy'}, {'ingredient': 'Spices and Condiments', 'description': 'Spices and condiments refer to a variety of flavoring substances used to enhance the taste of food. Examples include pepper, cinnamon, and garlic.', 'emoji': '🌶️', 'label': 'Healthy'}, {'ingredient': 'Acidity Regulators (INS 330, INS 296)', 'description': 'Acidity regulators are food additives used to control the acidity or alkalinity of a food product. INS 330 refers to citric acid, while INS 296 refers to malic acid.', 'emoji': '🔅', 'label': 'Healthy'}, {'ingredient': 'Sugar', 'description': 'Sugar is a sweet, crystalline substance extracted from sugarcane or sugar beets. It is commonly used as a sweetener in food and beverages.', 'emoji': '🍬', 'label': 'Unhealthy'}, {'ingredient': 'Raising Agent (INS 500(ii))', 'description': 'Raising agents are substances used in baking to help dough or batter rise. INS 500(ii) refers to sodium bicarbonate, also known as baking soda.', 'emoji': '🥐', 'label': 'Healthy'}, {'ingredient': 'Turmeric Powder', 'description': 'Turmeric powder is a bright yellow spice made from the dried root of the turmeric plant. It is commonly used in Indian and Southeast Asian cuisines.', 'emoji': '🌕', 'label': 'Healthy'}, {'ingredient': 'Citric Acid', 'description': 'Citric acid is a weak organic acid found in citrus fruits. It is commonly used as a flavoring agent and preservative in food and beverages.', 'emoji': '🍋', 'label': 'Healthy'}, {'ingredient': 'Tartrazine (INS 102)', 'description': 'Tartrazine, also known as FD&C Yellow No. 5, is a synthetic yellow dye commonly used in food and beverages. It may cause allergic reactions in some individuals.', 'emoji': '🟡', 'label': 'Unhealthy'}, {'ingredient': 'Allura Red (INS 129)', 'description': 'Allura Red, also known as FD&C Red No. 40, is a synthetic red dye commonly used in food and beverages. It may cause allergic reactions in some individuals.', 'emoji': '🔴', 'label': 'Unhealthy'}, {'ingredient': 'Paprika Extract (INS 160c)', 'description': 'Paprika extract is a natural food coloring derived from dried and ground red peppers. It is commonly used to add color and flavor to food products.', 'emoji': '🌶️', 'label': 'Healthy'}]}

print(x['ingredients'][0])
