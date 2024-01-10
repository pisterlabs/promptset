from llm_commons.langchain.btp_llm import init_llm


def extract_keywords_from_abstract(content, file_path):

#    llm = init_llm('gpt-35-turbo', temperature=0., max_tokens=256)
    llm = init_llm('gpt-35-turbo', temperature=0.5, max_tokens=256)

    type(llm)

    from langchain import LLMChain, PromptTemplate

    PROMPT = """Generate a list of keywords for the following text:
    
    ```markdown 
    For the following text give the 3-5 most important keywords.
    The keywords should be different and cover the whole topic, the keywords should be in order of importance, the keywords should reflect commons terms in the broader space of Software Engineering, Software Development and Software Architecture.
    The list of keywords should be in the form of a yaml array
        
    text: {text}
    ```
    """

    PROMPT = """Instructions: For the following text, give 3-5 keywords that best describe its main idea and content. The keywords should:    
        - Be different from each other and not repeat the same information.
        - Be specific and not too broad or too narrow.
        - Be representative of the text and the topic of software engineering, development, and architecture.
        - Be ordered from most to least important.
        - Be formatted as a yaml array starting.
        - The result should be an array of keywords in the form of a yaml array.   

        Text: {text}

        **Example:**
            Input text: Software engineering is the systematic application of engineering approaches to the development of software. Software engineering is a direct sub-discipline of engineering and has an overlap with computer science and management science. It is also considered a part of overall systems engineering.
            ['software engineering', 'development', 'engineering approaches', 'systems engineering', 'computer science']"""

    prompt = PromptTemplate.from_template(PROMPT)

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)




    keywords = llm_chain.run(content)
    print(keywords)
    return keywords

#--------- Usage ----------#
with open('ExampleAbstractsFromEcosystem/AI_Tooling.md') as stream:
   keywords = extract_keywords_from_abstract(stream, 'ExampleAbstractsFromEcosystem/AI_Tooling.md')





