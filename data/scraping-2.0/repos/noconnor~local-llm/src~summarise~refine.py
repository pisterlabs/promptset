from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


def summarise(llm, docs):
    #
    # This code explicitly shows the map and reduce prompts.
    # It can be simplified to just `chain = load_summarize_chain(llm, chain_type="refine")`
    #
    # Each doc (page) is summarised, then the summary and the next page are passed to the model to refine the summary
    # further, until there are no more pages
    # https://python.langchain.com/docs/use_cases/summarization#option-3.-refine
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary."
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,  # Set to true to show intermediate summaries
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"input_documents": docs}, return_only_outputs=True)
    return result["output_text"]
