from langchain import OpenAI, PromptTemplate


def getTextSemantics(text):
    llm = OpenAI(temperature=0.3)
    template = """
    You are a text expert who can understand the semantics of text  well. 
    Description of the text's semantics in around 50 words:
    {text}
    Your output needs to be in JSON format and include one fields.
    semantics: text semantics
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    return llm(prompt.format(text=text))
