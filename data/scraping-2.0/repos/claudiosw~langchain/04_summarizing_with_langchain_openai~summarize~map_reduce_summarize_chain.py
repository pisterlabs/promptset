from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_base import llm, get_chunks_from_text


def get_summary_using_map_reduce(text):
    chunks = get_chunks_from_text(text)

    chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        verbose=False
    )
    output_summary = chain.run(chunks)

    return output_summary


def get_summary_using_map_reduce_with_template(text):
    chunks = get_chunks_from_text(text)

    # Prompt template that is used to generate the summary of each chunk
    map_prompt = '''
    Write a short and concise summary of the following:
    Text: `{text}`
    CONCISE SUMMARY:
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )

    # Prompt template that is used to generate the final summary that combines the summary of each chunk
    combine_prompt = '''
    Write a concise summary of the following text that covers the key points.
    Add a title to the summary.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
    Text: `{text}`
    '''
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    output = summary_chain.run(chunks)
    return output
