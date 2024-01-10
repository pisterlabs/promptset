import os
from openai import AzureOpenAI
import getClothingSuggestionFromGpt
import sys

criteria = "gender = 'man', ethnicity = 'Moslem', age = '20s', destination = 'Casablanca', minTemp = 15, maxTemp = 35, minPrec = 0, maxPrec = 5, sunnyDays = 12"
suggestedEquipment = getClothingSuggestionFromGpt.getRecommendations(criteria)

print(suggestedEquipment)