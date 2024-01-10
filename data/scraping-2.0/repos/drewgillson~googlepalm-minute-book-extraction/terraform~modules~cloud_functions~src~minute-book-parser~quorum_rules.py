import main
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts quorum rules for directors and shareholders from the sorted pages of minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains the page number and file name of a sorted file.

    Returns:
        A list of dictionaries where each dictionary contains the extracted quorum details that match the minute book
        extraction schema. Each quorum object includes the date, extracted quorum rules, and the URL of the source
        document page where the rules were extracted from.
    """

    quorum_rules = []
    quorum_content = ""
    quorum_token_count = 0
    quorum_max_token_limit = 3072
    extracting_quorum = False

    file_count = len(sorted_files)
    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()

        if "quorum" in lowercase_content:
            extracting_quorum = True

        #  "directors_quorum": string, // Quorum rules for directors
        #  "shareholders_quorum": string, // Quorum rules for shareholders
        if extracting_quorum is True:
            quorum_tokens = quorum_token_count + main.num_tokens_from_string(content)
            shareholders_quorum, directors_quorum = [None, None]

            if quorum_tokens < quorum_max_token_limit:
                quorum_token_count += quorum_tokens
                quorum_content += content
                parsed_this_page = True

            # Quorum rules can sometimes be split across multiple pages so we need a larger context window
            if quorum_tokens >= quorum_max_token_limit or page_number == file_count:
                if parsed_this_page:
                    shareholders_quorum = extract_shareholders_quorum(quorum_content)
                    directors_quorum = extract_directors_quorum(quorum_content)
                else:
                    if shareholders_quorum is None:
                        shareholders_quorum = extract_shareholders_quorum(content)
                    if directors_quorum is None:
                        directors_quorum = extract_directors_quorum(content)

                quorum_rules.append({"directors_quorum": directors_quorum, "provenance": main.get_url(file_name)})
                quorum_rules.append({"shareholders_quorum": shareholders_quorum, "provenance": main.get_url(file_name)})

                extracting_quorum = False
                quorum_content = ""
                quorum_token_count = 0
                quorum_tokens = 0

    return quorum_rules


# The following functions use a large language model to perform question & answer-style extraction from a minute book


def extract_directors_quorum(content, entity_name):
    prompt = PromptTemplate(
        input_variables=["content", "entity_name"],
        template="""What constitutes quorum for meetings of directors of {entity_name} where only
                    one director is present? How about when two or more directors are present? Is
                    a majority of directors required for quorum? Explain in a concise paragraph.
                    THINK: Do not explain quorum for meetings of shareholders, this is irrelevant.
                    Passage:
                    {content}
                    Director Quorum:""")

    directors_quorum_candidate = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.5, max_output_tokens=512),
                                          prompt=prompt)

    return directors_quorum_candidate.predict(content=content).strip()


def extract_shareholders_quorum(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""What constitutes quorum for meetings of shareholders according to this passage?
                    THINK: Do not get confused between meetings of directors and meetings of shareholders.
                    Passage:
                    {content}
                    Shareholder Quorum:""")

    shareholders_quorum_candidate = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.5, max_output_tokens=512),
                                             prompt=prompt)

    return shareholders_quorum_candidate.predict(content=content).strip()
