
#query using the stuff summarization chain
def run(query:str):
    #input: the query question
    #output: summary based on the query
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI
    
    db = Chroma(persist_directory="./db/arkdb", embedding_function=OpenAIEmbeddings())
    res = db.similarity_search(query, k = 8)
    
    # Define prompt
    prompt_template = """Write a summary about """ + query + """ 
    Based on the text:
    "{text}"
    
    At the start of each section, provide reference based on the <stage> from source text, in the format of
    Example:
    In <stage>, <what happened 1>\n\n
    In <stage>, <what happened 2>\n\n
    ...
    
    REPLACE the <stage> with the actual stages from the text source in the output!
    
    Output:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Define LLM chain
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )
    return(stuff_chain.run(res))