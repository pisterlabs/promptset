import ai21
from langchain import LLMChain
# from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models.openai import ChatOpenAI
from langchain import PromptTemplate


def answer(text: str, summaries: str, context: str):
    # Create LLMChain with custom prompt for translation
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    prompt = get_prompt_for_predict()
    chain = LLMChain(llm=llm, prompt=prompt)

    # Return the translated text by running the prompt with the given input variables
    return chain.predict(query=text, context=context, summaries=summaries)

def get_prompt_for_predict() -> PromptTemplate:
    template = """The following query: {query} must be answered with the information specified in the user documents.
    These are the summaries of the provided documents:{summaries}
    This is the information relevant to the query found inside the documents:{context}
    Using this information write a long and detailed answer. If the provided context doesnt have the answer dont try to make it up, just say you dont know.
    """
    input_variables = ['context', 'summaries', 'query']

    return PromptTemplate(
        input_variables=input_variables,
        template=template
    )
