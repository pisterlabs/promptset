# -*- coding: utf-8 -*-
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.9)
text = "Какое название будет хорошим для компании, которая производит разноцветные носки?"
print(llm(text))
