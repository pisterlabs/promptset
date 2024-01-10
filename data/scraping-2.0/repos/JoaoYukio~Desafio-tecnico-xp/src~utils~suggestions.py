from langchain import PromptTemplate

prompt_template = """
    Gostaria que criasse {num_perguntas} perguntas interessantes sobre o texto {text} que você acabou de ler.
"""

prompt = PromptTemplate.from_template(prompt_template)
