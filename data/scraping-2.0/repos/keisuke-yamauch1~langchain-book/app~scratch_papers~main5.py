from langchain.prompts import PromptTemplate

if __name__ == "__main__":
    template = """
    以下の料理のレシピを考えてください
    
    料理名：{dish}
    """

    prompt = PromptTemplate(
        input_variables=["dish"],
        template=template,
    )

    result = prompt.format(dish="カレー")
    print(result)
