from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_base import llm, chunk_data


def get_summary_using_refine(data):
    chunks = chunk_data(data)
    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        verbose=True
    )
    output_summary = chain.run(chunks)
    return output_summary


def get_summary_using_refine_with_template(data):
    chunks = chunk_data(data)
    prompt_template = """Write a concise summary of the following extracting the key information:
        Text: `{text}`
        CONCISE SUMMARY:
    """
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    refine_template = '''
        Your job is to produce a final summary.
        I have provided an existing summary up to a certain point: {existing_answer}.
        Please refine the existing summary with some more context below.
        ------------
        {text}
        ------------
        Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
        by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.

    '''
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_answer', 'text']
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False
    )
    output_summary = chain.run(chunks)
