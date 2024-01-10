from langchain import PromptTemplate, OpenAI, LLMChain


def story(text, api):
    template = """
    You are story teller.
    You can narrate a story from the given context. The story shouldn't be more than 60 words. 
    The story should be interesting and heart warming or emotional or joyful.
    CONTEXT: {text}
    STORY:
"""
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_model = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",
                         temperature=1, openai_api_key=api), prompt=prompt, verbose=True)
    scene = llm_model.predict(text=text)
    return scene
