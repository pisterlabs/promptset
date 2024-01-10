import main
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts restrictions and provisions related to a corporation from a minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains the page number and file name of a sorted file.

    Returns:
        A list of dictionaries where each dictionary represents a set of restrictions or provisions and includes the date
        when they were established, the type of restriction or provision, the text of the restriction or provision, and the URL
        of the source document page where the restriction or provision was extracted from.
    """

    restrictions_provisions = []

    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()

        #  "transfer_restrictions": string, // Provisions or rules that limit or regulate the transfer or sale of a company's shares or other ownership interests
        if "transfer" in lowercase_content and "restrictions" in lowercase_content and "certificate" not in lowercase_content:
            output = extract_transfer_restrictions(content)
            restrictions_provisions.append({"transfer_restrictions": output, "provenance": main.get_url(file_name)})

        #  "other_restrictions": string, // Restrictions on the corporation's activities
        if "other" in lowercase_content and "restrictions" in lowercase_content and "certificate" not in lowercase_content:
            output = extract_other_restrictions(content)
            restrictions_provisions.append({"other_restrictions": output, "provenance": main.get_url(file_name)})

        #  "other_provisions": string, // Additional provisions or rules that are not covered by the other properties
        if "other provisions" in lowercase_content:
            output = extract_other_provisions(content)
            restrictions_provisions.append({"other_provisions": output, "provenance": main.get_url(file_name)})

    return restrictions_provisions


# The following functions use a large language model to perform question & answer-style extraction from a minute book


def extract_other_restrictions(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""If this passage from a set of corporate by-laws
                    pertains to other restrictions, read the restrictions and then describe
                    them concisely. Do not include share transfer restrictions. Do not include
                    information about the minimum or maximum number of directors. Format output
                    as a single line without linebreaks.
                    Passage:
                    {content}
                    Other Restrictions:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2,
                                    max_output_tokens=512),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "Not Found":
        return output


def extract_transfer_restrictions(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""If this passage from a set of corporate by-laws
                    pertains to share transfer restrictions, read the restrictions and then
                    describe them concisely. Do not include any other restrictions except
                    for share transfer restrictions. Do not include information about the
                    minimum or maximum number of directors. Format output as a single line
                    without linebreaks.
                    Passage:
                    {content}
                    Share Transfer Restrictions:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2,
                                  max_output_tokens=512),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "Not Found":
        return output


def extract_other_provisions(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""If this passage from a set of corporate by-laws pertains to other provisions,
                    read the provisions and then describe them. Do not include information about
                    the minimum or maximum number of directors. Format output as a single line
                    without linebreaks.
                    Passage:
                    {content}
                    Other Provisions:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2,
                                  max_output_tokens=512),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "Not Found":
        return output
