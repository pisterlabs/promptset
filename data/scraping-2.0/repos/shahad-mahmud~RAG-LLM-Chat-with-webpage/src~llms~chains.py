from langchain.chains import LLMChain


def get_LLMChain(prompt, llm):
    chain = LLMChain(
        prompt=prompt,
        llm=llm,
    )

    return chain
