from langchain import PromptTemplate
from text_generation import openai, davinci

pt_gen = PromptTemplate(
    input_variables=["statement"],
    template="""Human: Please follow these steps:
1. Write three topic sentences arguing for {statement}.
2. Write three topic sentences arguing against {statement}.
3. Write an essay by expanding each topic sentence from Steps 1 and 2, and adding a conclusion to synthesize the arguments. Please enclose the essay in <essay></essay> tags.

Assistant:
    """)

print(openai(pt_gen.format(statement="4th law of thermodynamics importance")))
print(davinci(pt_gen.format(statement="flat earth")))
