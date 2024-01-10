from .pdf_scrape import query_pinecone_chain

def return_answer_and_sections(query):
    answer, docs = query_pinecone_chain(query)
    highway_code_sections = return_highway_sections(docs)
    return answer, highway_code_sections

def return_highway_sections(documents):
    sections = []

    for document in documents:
        # Split the page_content by the word "Rule" to separate sections
        section_parts = document.page_content.split('Rule ')

        # Skip the first part, which is typically empty
        for section_part in section_parts[1:]:
            # Extract the section title and content
            section_lines = section_part.split('\n', 1)
            if len(section_lines) == 2:
                section_title = f"Rule {section_lines[0]}"  # Re-add "Rule" to the section title
                section_content = section_lines[1]

                # Create a dictionary to store the section title and content
                section = {
                    'section_title': section_title,
                    'section_content': section_content,
                }

                sections.append(section)
    
    return sections

# Now, the `sections` list contains dictionaries with section titles and content

if __name__ == '__main__':
    import os
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores import Pinecone
    import pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    # OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    PINECONE_ENV = os.environ["PINECONE_ENV"]

    # Load the PDF using PyPDFLoader
    pdf_path = 'the_official_highway_code.pdf'
    loader = PyPDFLoader(pdf_path)

    pinecone.init(
        api_key='bc4d9f6e-c16b-482c-ac7e-8b090b656816',  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    from keys import OPENAI_API_KEY
    index_name = "highway-code" # put in the name of your pinecone index here
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain
    llm = OpenAI(temperature=1, openai_api_key=OPENAI_API_KEY)
    from langchain.prompts import PromptTemplate

    template = """
        You are assessing an automotive car accident claim. Please provide guidance on who, if anyone, is at fault. The context will include relevant parts of the highway code.
        Context: {context}
        Question: {question}
        If the answer is not in the context, DO NOT MAKE UP AN ANSWER.
        However, in this case, if there are any relevant answers you can find, please state these. 

    """
    prompt = PromptTemplate.from_template(
            template
        )
    query = '''
    Based on the information provided, here is the order of events in the car accident:

    The user was driving on the main road an hour ago.
    Another vehicle pulled out from the right side of the user near a set of traffic lights.
    The user was driving at 30mph and was worried about missing their appointment.
    There were no injuries reported.
    The user's car sustained damage on the right side and the wheel is making a strange noise.
    The user's car is not drivable.

    '''
    docs = docsearch.similarity_search(query)
    sections = return_highway_sections(docs)
    # all_context = ('').join([f.page_content for f in docs])
    # final_prompt = prompt.format(context=all_context ,question=query)
    # answer = llm(final_prompt)
    # print("answer: ", answer)
    # print("docs: ",docs)





