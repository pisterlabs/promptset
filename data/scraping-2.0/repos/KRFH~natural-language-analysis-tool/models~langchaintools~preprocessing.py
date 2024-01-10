from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from flask import session


def chat_tool_with_pandas_df(df, query):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Translate the following Japanese text to English: {text}",
    )
    chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=session["api_key"]), prompt=prompt)
    translated_query = chain.run(query)
    # translated_query = translate_to_english(query)
    translated_query += "The operation must be applied directly to the original dataframe."
    translated_query += "The answer must be just python code"
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, openai_api_key=session["api_key"]),
        df,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
    )
    result = agent.run(translated_query)
    print(query)
    print(translated_query)
    print(result)
    if "inplace" in result:
        pass
    elif " = " in result:
        exec(result)
    else:
        df = eval(result)

    return df
