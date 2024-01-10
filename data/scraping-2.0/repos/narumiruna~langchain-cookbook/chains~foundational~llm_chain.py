from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from loguru import logger


def main():
    """https://langchain-langchain.vercel.app/docs/modules/chains/foundational/llm_chain"""
    load_dotenv(find_dotenv())

    llm = OpenAI(temperature=0, verbose=True)
    logger.info('llm: {}', llm)

    prompt_template = 'What is a good name for a company that makes {product}?'
    prompt = PromptTemplate.from_template(prompt_template)
    logger.info('prompt: {}', prompt)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    logger.info('llm_chain: {}', llm_chain)

    llm_result = llm_chain('colorful socks')
    logger.info('llm_result: {}', llm_result)

    llm_result = llm_chain.predict(product='colorful socks')
    logger.info('llm_result: {}', llm_result)


if __name__ == '__main__':
    main()
