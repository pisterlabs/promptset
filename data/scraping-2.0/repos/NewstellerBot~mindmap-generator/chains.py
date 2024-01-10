from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import reduce_template, map_template, mindmap_template


def setup_chains(openai_api_key: str) -> (MapReduceDocumentsChain, LLMChain):
    """Setup the chains"""

    # Create the chat model
    llm = ChatOpenAI(
        temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k"
    )

    map_chain = LLMChain(llm=llm, prompt=map_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_template)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=16000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    mindmap_chain = LLMChain(llm=llm, prompt=mindmap_template)

    return map_reduce_chain, mindmap_chain
