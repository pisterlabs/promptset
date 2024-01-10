from abc import ABC, abstractmethod
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import AzureChatOpenAI
from functools import cached_property
from typing import Sequence
from langchain.schema import HumanMessage, BaseMessage
import pandas as pd
from io import StringIO


sample_healthy_foods = [
    'walnuts', 'lentils', 'kale', 'blueberries', 'potatoes',
    'plain (0%) greek yogurt', 'avocado', 'sardines', 'eggs', 'carrots'
]

query = f"""For all the items in the list: {','.join(sample_healthy_foods)}

build the table with the following form:

Food|Quantity unit|Calories per 100 gram|Carbs (g)|Fat (g)|Protein (g)|Fiber (g)|Sugars total including NLEA|Calcium|Iron|Magnesium|Phosphorus|Potassium|Sodium|Zinc|Copper|Manganese|Selenium|Vitamin C|Thiamin|Riboflavin|Niacin|Pantothenic acid|Vitamin B-6|Folate|Choline|Betaine|Vitamin B-12|Vitamin A RAE|Retinol|Carotene beta|Carotene alpha|Cryptoxanthin beta|Vitamin A IU|Lycopene|Lutein + zeaxanthin|Vitamin E (alpha-tocopherol)|Vitamin D (D2 + D3) International Units|Vitamin K (phylloquinone)|Fatty acids total saturated|Fatty acids total monounsaturated|Fatty acids total polyunsaturated|Fatty acids total trans|Cholesterol|Tryptophan|Threonine|Isoleucine|Leucine|Lysine|Methionine|Cystine|Phenylalanine|Tyrosine|Valine|Arginine|Histidine|Alanine|Aspartic acid|Glutamic acid|Glycine|Proline|Serine|Alcohol ethyl|Caffeine|Theobromine
Carrots 90.00   g       41      10      0.2     0.9     2.8 ...

Do not show the code just show the markdown table with all the nutritional information. In the output, skip the header of the table."""


class LLM(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def model(self) -> BaseChatModel:
        pass

    def answer(self, query: str) -> str:
        return self.model.predict(query)

class GPT4(LLM):

    @property
    def name(self) -> str:
        return "GPT-4"

    @cached_property
    def model(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(deployment_name="gpt-4")


def get_data(query: str = query, filename: str = "data/gpt_nutrition.csv", save=True):
    llm = GPT4()
    result = llm.answer(query)

    # parsing
    table = pd.read_table(StringIO(result), sep="|")
    table.columns = list(table.columns[2:]) + ['', '']
    table = table.iloc[1:, :65]

    # saving
    if save:
        table.to_csv(filename, index=False)

    return result, table
