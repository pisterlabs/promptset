from langchain.vectorstores import FAISS

from typing import Any, Dict, List


def clean_key_id(string: str, key_id: str):
    f"""
    Clean key_id number from string. 
    """
    string = string.replace(key_id, "")
    return string.strip()


def clean_pcs(pcs: str):
    f"""
    Clean Pcs related data.
    """
    s = "".join([x for x in pcs if x in {'1234567890'}])
    if len(s) == 0 or len(s) > 2:
        return 'NA'
    return int(s)


def clean_wts(wts: str):
    f"""
    Clean Pcs related data.
    """
    s = "".join([ x for x in wts if x in {'1234567890'}])
    if len(s) == 0 or len(s) > 3:
        return 'NA'
    return int(s)


def cleanup(by: str, thing: str):
    f"""
    Clean function to use based on object tag.
    """
    fns = {
        "Pcs": clean_pcs,
        "Wts": clean_wts,
    }
    if by not in fns:
        return thing
    return fns[by](thing)


def knowledge_graph(
    key_id: str,
    docsearch: FAISS,
    pdf_inputs: List[str],
    query: str,
    rules_template: str,
    chain: Any,
):
    f"""
    Build knowledge graph for relationships
    between tags and data in PDF documents.
    """
    
    main_data: Dict[str, Any] = dict()

    for pdf_input in pdf_inputs:
        # General extraction
        prompt = query + pdf_input + ". " + rules_template
        docs = docsearch.similarity_search(prompt)
        response = chain.run(input_documents=docs, question=prompt)
        if response == "I don't know.":
            response = ""
        if pdf_input == key_id:
            for num in response.split("\n"):
                if len(num) > 0:
                    main_data[num] = {}
        else:
            indiv_responses = response.split("\n")
            indiv_responses = [x for x in indiv_responses if len(x) > 2]
            if not indiv_responses:
                for num in main_data:
                    main_data[num][pdf_input] = indiv_responses
            else:
                for indiv_response in indiv_responses:
                    for num in main_data:
                        if num in indiv_response:
                            resp = clean_key_id(indiv_response, num)
                            resp = cleanup(by=pdf_input, thing=resp)
                            main_data[num][pdf_input] = resp
        # Ensure data is returned despite no collection
        for num in main_data:
            if pdf_input != key_id and pdf_input not in main_data[num]:
                main_data[num][pdf_input] = "NA"
    return main_data