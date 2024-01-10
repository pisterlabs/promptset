from langchain.output_parsers import CommaSeparatedListOutputParser

recipe_output_parser = CommaSeparatedListOutputParser()

# RECIPE_EXTRACTOR_PROMPT = """
# Act as an expert in extracting recipes from text that understand danish and english.
# Given an unstructured text containing a recipe, extract the ingredients including the amounts of the ingredients (quantity, weight etc.).
# Sometimes, there is no recipe to be found and then you return an empty python list ([]).

# Sometimes the recipe is already provided. In that case just output the input in the format described below.

# The input/text is delimited by ####.

# {format_instructions}

# Begin!

# ####
# {input}
# ####
# """

RECIPE_EXTRACTOR_PROMPT = """
Act as an expert in extracting recipes from text that understand danish and english.
Given an unstructured text containing a recipe, extract the ingredients and the amount of the recipe.
Sometimes, there is no recipe to be found and then you return 'No ingredients'.


Sometimes the ingredients list is already provided. In that case just output the input in the format described below.

Example of ingredients already provided in Danish: oksemørbrad (250 g), 2 gulerødder
Example of ingredients already provided in English:
250 g cream
400 g beef tenderloin

The input/text is delimited by ####.

{format_instructions}

Begin!

####
dansk hovedret 12 tilberedningstid 45 minutter arbejdstid 25 minutter print bedøm denne opskrift rated 4
/ 5 based on 1 customer reviews hov! du skal være logget ind. log ind bliv medlem ingredienser (12) 1 2 3 4 5 6 7 8
antal personer: 500 gram torskefilet 1 tsk havsalt 2 stk æg 1 stk gulerod 0.5 deciliter fløde 13% 0.5 tsk revet
muskatnød 1 tsk peber 2 spsk olie 4 deciliter creme fraiche 18% 4 stk æggeblomme 2 spsk frisk dild 4 spsk frisk persille
####

500 gram torskefilet, 1 tsk havsalt, 2 stk æg, 1 stk gulerod, 0.5 deciliter fløde 13%, 0.5 tsk revet muskatnød, 1 tsk peber, 2 spsk olie, 4 deciliter creme fraiche 18%, 4 stk æggeblomme, 2 spsk frisk dild, 4 spsk frisk persille

####
{input}
####
"""
