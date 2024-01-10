from langchain import PromptTemplate

from estimator.output_parsers.sql_co2_estimator import sql_co2_output_parser

EN_LANGUAGE = "English"
DK_LANGUAGE = "Danish"

EN_EXAMPLE_QUERY = (
    "'SELECT Name, Total_kg_CO2_eq_kg FROM dk_co2_emission WHERE Name LIKE '%tomato%' OR Name LIKE '%bouillon%'"
)
DK_EXAMPLE_QUERY = (
    "'SELECT Navn, Total_kg_CO2_eq_kg FROM dk_co2_emission WHERE Navn LIKE '%tomat%' OR Navn LIKE '%bouillion%'"
)

EN_EXAMPLE_REMOVING = (
    "'SELECT Navn, Total_kg_CO2_eq_kg FROM dk_co2_emission WHERE Navn LIKE '%tomato%' OR Navn LIKE '%bouillion%'"
)
DK_EXAMPLE_REMOVING = "'%hakkede tomater%' to '%tomat%' or '%hakket oksekød%' to '%oksekød%'"

EN_EXAMPLE_MATCH = "'1 can of chopped tomatoes' best matches results from 'Tomato, peeled, canned'."
DK_EXAMPLE_MATCH = "'1 dåse hakkede tomater' best matches results from 'Tomat, flået, konserves'."

# EN_EXAMPLE_ANSWER_FOUND = "'Chopped tomatoes: X kg CO2e / kg \n'"
# DK_EXAMPLE_ANSWER_FOUND = "'Hakkede tomater: X kg CO2e/ kg \n'."

# EN_EXAMPLE_ANSWER_NOT_FOUND = "'Chopped tomatoes: ? \n'"
# DK_EXAMPLE_ANSWER_NOT_FOUND = "'Hakkede tomater: ? \n'."

EN_INGREDIENTS_EXAMPLE = """
150 g red lentils
1 can of chopped tomatoes
2 cubes of vegetable bouillon
1 tin of tomato concentrate (140 g)
1 tbsp. lemon juice
1. tbsp. chili powder
1 starfruit
"""

DK_INGREDIENTS_EXAMPLE = """
150 g røde linser
1 dåse hakkede tomater
2 terninger grøntsagsbouillon
1 dåse tomatkoncentrat (140 g)
1 spsk. citronsaft
1. spsk. chilipulver
10 majstortillas
1 stjernefrugt
"""

EN_SQL_QUERY_EXAMPLE = """
SELECT Name, Total_kg_CO2_eq_kg FROM dk_co2_emission WHERE
          Name LIKE '%tomato%' OR
          Name LIKE '%lentil%' OR
          Name LIKE '%bouillon%' OR
          Name LIKE '%juice%' OR
          Name LIKE '%lemon%' OR
          Name LIKE '%chili%' OR
          Name LIKE '%starfruit%'
"""

DK_SQL_QUERY_EXAMPLE = """
SELECT Navn, Total_kg_CO2_eq_kg FROM dk_co2_emission WHERE
          Navn LIKE '%tomat%' OR
          Navn LIKE '%linse%' OR
          Navn LIKE '%bouillon%' OR
          Navn LIKE '%saft%' OR
          Navn LIKE '%citron%' OR
          Navn LIKE '%chili%' OR
          Navn LIKE '%tortilla%' OR
          Navn LIKE '%stjernefrugt%'
"""

EN_SQL_RESULT_EXAMPLE = """
[('Tomato, ripe, raw, origin unknown', 0.7), ('Green lentils, dried', 1.78)
            ('Tomatojuice, canned', 1.26), ('Tomato, peeled, canned', 1.26)
            ('Tomato paste, concentrated', 2.48), ('Red lentils, dried', 1.78
            ('Ice, popsickle, lemonade', 1.15), ('Lemon, raw', 0.94
            ('Apple juice', 1.64),('Bouillon, chicken, prepared', 0.38)
            ('Bouillon, beef, prepared', 0.52), ('Pepper, hot chili, raw', 1.02)
            ('Pepper, hot chili, canned', 1.54),
            ]
"""

DK_SQL_RESULT_EXAMPLE = """
[('Tomat, uspec., rå', 0.7), ('Grønne linser, tørrede', 1.78)
            ('Tomatjuice, konserves', 1.26), ('Tomat, flået, konserves', 1.26)
            ('Tomatpure, koncentreret', 2.48), ('Røde linser, tørrede', 1.78
            ('Ispind, limonade', 1.15), ('Citron, rå', 0.94
            ('Æblejuice', 1.64),('Bouillon, hønsekød, spiseklar', 0.38)
            ('Bouillon, oksekød, spiseklar', 0.52), ('Peber, chili, rå', 1.02)
            ('Tortillabrød, hvede',0.74), ('Peber, chili, konserves', 1.54),
            ]
"""

EN_FINAL_ANSWER_EXAMPLE = """
{
  "emissions": [
    {
      "ingredient": "150 g red lentils",
      "comment": "",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.78
    },
    {
      "ingredient": "1 can of chopped tomatoes",
      "comment": "",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.26
    },
    {
      "ingredient": "2 cubes of vegetable bouillon",
      "comment": "closest was chicken bouillon",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 0.38
    },
    {
      "ingredient": "1 tin of tomato concentrate (140 g)",
      "comment": "closest was tomato paste",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 2.48
    },
    {
      "ingredient": "1 tbsp. lemon juice",
      "comment": "Closest was Lemon, raw",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 0.94
    },
    {
      "ingredient": "1. tbsp. chili powder",
      "comment": "closest was 'Pepper, hot chili, canned'",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.54
    },
    {
      "ingredient": "1 starfruit",
      "comment": "Not found in database",
      "unit": "kg CO2e / kg",
      "co2_per_kg": null
    }
  ]
}
"""

DK_FINAL_ANSWER_EXAMPLE = """
{
  "emissions": [
    {
      "ingredient": "150 g røde linser",
      "comment": "",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.78
    },
    {
      "ingredient": "1 dåse hakkede tomater",
      "comment": "",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.26
    },
    {
      "ingredient": "2 terninger grøntsagsbouillon",
      "comment": "tættest var Bouillon, hønsekød, spiseklar",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 0.38
    },
    {
      "ingredient": "1 dåse tomatkoncentrat (140 g)",
      "comment": "tættest var tomatpure, koncentreret",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 2.48
    },
    {
      "ingredient": "1 spsk. citronsaft",
      "comment": "Tættest var citron, rå",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 0.94
    },
    {
      "ingredient": "1. spsk. chilipulver",
      "comment": "tættest var Peber, chili, konserves",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 1.54
    },
    {
      "ingredient": "10 majstortillas",
      "comment": "Tættest var tortillabrød, hvede",
      "unit": "kg CO2e / kg",
      "co2_per_kg": 0.74
    },
    {
      "ingredient": "1 stjernefrugt",
      "comment": "Ikke fundet i databasen",
      "unit": "kg CO2e / kg",
      "co2_per_kg": null
    }
  ]
}
"""


CO2_SQL_PROMPT_TEMPLATE = """
Given a list of ingredients in {language}, extract the main ingredients from the list
and create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

Solve the task using the following steps:
- Query all ingredients in a single query. Make sure you query ALL the ingredients provided after `Ingredients:`
  Example query: {example_query}
- In the query, remove all non-ingredient words.
  Example of removing: {example_removing}
- Match the SQLResult to the list of ingredients based on preparation and type.
  Example match: {example_match}
- Return the Answer by the format instructions explained below.
- Do not provide any ranges for the final answer. For example, do not provide '0.1-0.5 kg CO2e per kg' as the final answer.
  Instead, return the closest match.

Use the following format:
Ingredients: "Ingredients here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"

Only use the following tables:
{table_info}

{format_instructions}

Begin!

Ingredients:
{ingredients_example}

SQLQuery: {query_example}

SQLResult: {query_result_example}

{final_answer_example}

Ingredients: {input}"""

EN_CO2_SQL_PROMPT_TEMPLATE = PromptTemplate(
    template=CO2_SQL_PROMPT_TEMPLATE,
    input_variables=["dialect", "table_info", "input"],
    partial_variables={
        "language": EN_LANGUAGE,
        "example_query": EN_SQL_QUERY_EXAMPLE,
        "example_removing": EN_EXAMPLE_REMOVING,
        "example_match": EN_EXAMPLE_MATCH,
        # "example_answer": EN_EXAMPLE_ANSWER_FOUND,
        # "example_not_found": EN_EXAMPLE_ANSWER_NOT_FOUND,
        "ingredients_example": EN_INGREDIENTS_EXAMPLE,
        "query_example": EN_SQL_QUERY_EXAMPLE,
        "query_result_example": EN_SQL_RESULT_EXAMPLE,
        "format_instructions": sql_co2_output_parser.get_format_instructions(),
        "final_answer_example": EN_FINAL_ANSWER_EXAMPLE,
    },
)

DK_CO2_SQL_PROMPT_TEMPLATE = PromptTemplate(
    template=CO2_SQL_PROMPT_TEMPLATE,
    input_variables=["dialect", "table_info", "input"],
    partial_variables={
        "language": DK_LANGUAGE,
        "example_query": DK_SQL_QUERY_EXAMPLE,
        "example_removing": DK_EXAMPLE_REMOVING,
        "example_match": DK_EXAMPLE_MATCH,
        # "example_answer": DK_EXAMPLE_ANSWER_FOUND,
        # "example_not_found": DK_EXAMPLE_ANSWER_NOT_FOUND,
        "ingredients_example": DK_INGREDIENTS_EXAMPLE,
        "query_example": DK_SQL_QUERY_EXAMPLE,
        "query_result_example": DK_SQL_RESULT_EXAMPLE,
        "format_instructions": sql_co2_output_parser.get_format_instructions(),
        "final_answer_example": DK_FINAL_ANSWER_EXAMPLE,
    },
)
