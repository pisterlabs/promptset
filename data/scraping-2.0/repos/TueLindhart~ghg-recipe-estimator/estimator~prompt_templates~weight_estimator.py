from langchain import PromptTemplate

from estimator.output_parsers.weight_estimator import weight_output_parser

EN_WEIGHT_RECALCULATIONS = """
1 can = 400 g = 0.4 kg
1 bouillon cube = 4 g = 0.004 kg
1 large onion = 285 g = 0.285 kg
1 medium onion = 170 g = 0.170 kg
1 small onion = 115 g = 0.115 kg
1 bell pepper = 150 g = 0.150 kg
1 can tomato paste = 140 g = 0.140 kg
1 tablespoon/tbsp. = 15 g  = 0.015 kg
1 teaspoon/tsp. = 5 g = 0.005 kg
1 potato = 170 - 300 g = 0.170 - 0.300 kg
1 carrot = 100 g = 0.100 kg
1 lemon = 85 g = 0.085 kg
1 tortilla = 30 g = 0.030 kg
1 squash = 400 g = 0.400 kg
1 clove garlic = 0.004 kg
1 dl / deciliter = 0.1 kg
Handful of herbs (basil, oregano etc.) = 0.025 kg

Examples of a bunch/bnch of an ingredient - use them as a guideline:
1 bunch/bnch parsley = 50 g = 0.050 kg
1 bunch/bnch asparagus = 500 g = 0.500 kg
1 bunch of carrots = 750 g = 0.750 kg
1 bunch/bnch tomatoes = 500 g = 0.500 kg
The weights of bunches are estimated as the highest possible weight.
"""

DK_WEIGHT_RECALCULATIONS = """
1 dåse = 400 g = 0.4 kg
1 terning bouillon = 4 g = 0.004 kg
1 stor løg = 285 g = 0.285 kg
1 mellem løg = 170 g = 0.170 kg
1 lille løg = 115 g = 0.115 kg
1 peberfrugt = 150 g = 0.150 kg
1 dåse tomatkoncentrat = 140 g = 0.140 kg
1 spiseskefuld/spsk. = 15 g  = 0.015 kg
1 teskefuld/tsk. = 5 g = 0.005 kg
1 kartoffel = 170 - 300 g = 0.170 - 0.300 kg
1 gulerod = 100 g = 0.100 kg
1 citron = 85 g = 0.085 kg
1 tortilla = 30 g = 0.030 kg
1 squash = 400 g = 0.400 kg
1 fed hvidløg = 0.004 kg
1 dl / deciliter = 0.1 kg
Håndful urter (basilikum, oregano osv.) = 0.025 kg

Examples of bdt/bundt af en ingrediens - use them as a guideline:
1 bundt/bdt persille = 50 g = 0.050 kg
1 bundt/bdt asparges = 500 g = 0.500 kg
1 bundt gulerødder = 750 g = 0.750 kg
1 bundt/bdt tomater = 500 g = 0.500 kg
The weights of bdt/bundt are estimated the highest possible weight.
"""

EN_INPUT_EXAMPLE = """
1 can chopped tomatoes
200 g pasta
500 ml water
250 grams minced meat
0.5 cauliflower
1 tsp. sugar
1 organic lemon
3 teaspoons salt
2 tbsp. spices
pepper
2 large potatoes
1 bunch asparagus
"""

DK_INPUT_EXAMPLE = """
1 dåse hakkede tomater
200 g pasta
500 ml vand
250 gram hakket kød
0.5 blomkål
1 tsk. sukker
1 økologisk citron
3 teskefulde salt
2 spsk. krydderi
peber
2 store kartofler
1 bdt asparges
"""

DK_ANSWER_EXAMPLE = """
{
  "weight_estimates": [
    {
      "ingredient": "1 dåse hakkede tomater",
      "weight_calculation": "1 dåse = 400 g = 0.4 kg",
      "weight_in_kg": 0.4
    },
    {
      "ingredient": "200 g pasta",
      "weight_calculation": "200 g = 0.2 kg",
      "weight_in_kg": 0.2
    },
    {
      "ingredient": "500 ml vand",
      "weight_calculation": "500 ml = 0.5 kg",
      "weight_in_kg": 0.5
    },
    {
      "ingredient": "250 gram hakket kød",
      "weight_calculation": "250 g = 0.25 kg",
      "weight_in_kg": 0.25
    },
    {
      "ingredient": "0.5 blomkål",
      "weight_calculation": "1 blomkål = 500 g (estimeret af LLM model) = 0.5 kg",
      "weight_in_kg": 0.5
    },
    {
      "ingredient": "1 tsk. sukker",
      "weight_calculation": "1 teskefuld = 5 g = 0.005 kg",
      "weight_in_kg": 0.005    },
    {
      "ingredient": "1 økologisk citron",
      "weight_calculation": "1 citron = 85 g = 0.085 kg",
      "weight_in_kg": 0.085
    },
    {
      "ingredient": "3 teskefulde salt",
      "weight_calculation": "1 tsk. = 5 g, 3 * 5 g = 15 g = 0.015 kg",
      "weight_in_kg": 0.015    },
    {
      "ingredient": "2 spsk. krydderi",
      "weight_calculation": "1 spsk. = 15 g, 2 * 15 g = 30 g = 0.030 kg",
      "weight_in_kg": 0.03    },
    {
      "ingredient": "peber",
      "weight_calculation": "antal peber er ikke angivet.",
      "weight_in_kg": null    },
    {
      "ingredient": "2 store kartofler",
      "weight_calculation": "1 stor kartoffel = 300 g, 2 * 300 g = 600 g = 0.6 kg",
      "weight_in_kg": 0.6
    },
    {
      "ingredient": "1 bdt asparges",
      "weight_calculation": "1 bdt asparges = 500 g = 0.500 kg",
      "weight_in_kg": 0.5
    }
  ]
}
"""

EN_ANSWER_EXAMPLE = """
{
  "weight_estimates": [
    {
      "ingredient": "1 can chopped tomatoes",
      "weight_calculation": "1 can = 400 g = 0.4 kg",
      "weight_in_kg": 0.4
    },
    {
      "ingredient": "200 g pasta",
      "weight_calculation": "200 g = 0.2 kg",
      "weight_in_kg": 0.2
    },
    {
      "ingredient": "500 ml water",
      "weight_calculation": "500 ml = 0.5 kg",
      "weight_in_kg": 0.5
    },
    {
      "ingredient": "250 grams minced meat",
      "weight_calculation": "250 g = 0.25 kg",
      "weight_in_kg": 0.25
    },
    {
      "ingredient": "0.5 cauliflower",
      "weight_calculation": "1 cauliflower = 500 g (estimated by LLM model) = 0.5 kg",
      "weight_in_kg": 0.5
    },
    {
      "ingredient": "1 tsp. sugar",
      "weight_calculation": "1 teaspoon = 5 g = 0.005 kg",
      "weight_in_kg": 0.005    },
    {
      "ingredient": "1 organic lemon",
      "weight_calculation": "1 lemon = 85 g = 0.085 kg",
      "weight_in_kg": 0.085
    },
    {
      "ingredient": "3 teaspoons salt",
      "weight_calculation": "1 tsp. = 5 g, 3 * 5 g = 15 g = 0.015 kg",
      "weight_in_kg": 0.015    },
    {
      "ingredient": "2 tbsp. spices",
      "weight_calculation": "1 tbsp. = 15 g, 2 * 15 g = 30 g = 0.030 kg",
      "weight_in_kg": 0.03    },
    {
      "ingredient": "pepper",
      "weight_calculation": "amount of pepper not specified",
      "weight_in_kg": null    },
    {
      "ingredient": "2 large potatoes",
      "weight_calculation": "1 large potato = 300 g, 2 * 300 g = 600 g = 0.6 kg",
      "weight_in_kg": 0.6
    },
    {
      "ingredient": "1 bunch asparagus",
      "weight_calculation": "1 bunch asparagus = 500 g = 0.500 kg",
      "weight_in_kg": 0.5
    }
  ]
}
"""

WEIGHT_EST_PROMPT = """
Given a list of ingredients, estimate the weights in kilogram for each ingredient.
Explain your reasoning for the estimation of weights.

The following general weights can be used for estimation:
{recalculations}

If an ingredient is not found in the list of general weights, try to give your best estimate
of the weight in kilogram/kg of the ingredient and say (estimated by LLM model).
Your estimate must always be a python float. Therefore, you must not provide any intervals.

Input is given after "Ingredients:"

{format_instructions}

Ingredients:
{input_example}

Answer:
{answer_example}

Ingredients:
{input}
"""


DK_WEIGHT_EST_PROMPT = PromptTemplate(
    template=WEIGHT_EST_PROMPT,
    input_variables=["input"],
    partial_variables={
        "recalculations": DK_WEIGHT_RECALCULATIONS,
        "input_example": DK_INPUT_EXAMPLE,
        "answer_example": DK_ANSWER_EXAMPLE,
        "format_instructions": weight_output_parser.get_format_instructions(),
    },
)

EN_WEIGHT_EST_PROMPT = PromptTemplate(
    template=WEIGHT_EST_PROMPT,
    input_variables=["input"],
    partial_variables={
        "recalculations": EN_WEIGHT_RECALCULATIONS,
        "input_example": EN_INPUT_EXAMPLE,
        "answer_example": EN_ANSWER_EXAMPLE,
        "format_instructions": weight_output_parser.get_format_instructions(),
    },
)
