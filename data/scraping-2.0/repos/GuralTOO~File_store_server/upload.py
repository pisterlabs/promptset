from dotenv import load_dotenv
import openai
import WeaviateClient
import os
import SupabaseClient
from load_pdf import load_pdf_with_textract as load_pdf


class_name = "File_store_ver2"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_context_for_authors(properties=[""], k=3, path=""):
    properties.append("path")
    properties.append("page_number")
    pathFilter = {"path": "path", "operator": "Equal", "valueString": path}
    page_filter = {"path": "page_number",
                   "operator": "Equal", "valueString": "1"}
    client = WeaviateClient.get_client()
    text_query = "Can you return the names of the authors of the paper?"
    results = (
        client.query
        .get(class_name=class_name, properties=properties)
        .with_where({
            "operator": "And",
            "operands": [pathFilter, page_filter]
        })
        .with_near_text({"concepts": text_query})
        .with_limit(k)
        .do()
    )

    search_result = ""
    for i in range(len(results["data"]["Get"][class_name])):
        search_result += results["data"]["Get"][class_name][i][properties[0]] + ".\n"

    return search_result


def get_context_for_methods(properties=[""], k=3, path=""):
 
    properties.append("path")
    pathFilter = {"path": "path", "operator": "Equal", "valueString": path}
    client = WeaviateClient.get_client()
    text_query = "Can you provide the research methods, procedures, experimental protocols, practical applications, and detailed methodology described in the paper?"
    results = (
        client.query
        .get(class_name=class_name, properties=properties)
        .with_where(pathFilter)
        .with_near_text({"concepts": text_query})
        .with_limit(k)
        .do()
    )

    search_result = ""
    for i in range(len(results["data"]["Get"][class_name])):
        search_result += results["data"]["Get"][class_name][i][properties[0]] + ".\n"

    return search_result


def get_context_for_key_results(properties=[""], k=3, path=""):
    print("path in results: ", path)
    
    properties.append("path")
    pathFilter = {"path": "path", "operator": "Equal", "valueString": path}
    client = WeaviateClient.get_client()
    text_query = "Can you provide the main results, discussion, outcomes, conclusions, and findings described in the paper?"
    
    results = (
        client.query
        .get(class_name=class_name, properties=properties)
        .with_where(pathFilter)
        .with_near_text({"concepts": text_query})
        .with_limit(k)
        .do()
    )
    print("path after filter: ", path)
    print()
    search_result = ""
    for i in range(len(results["data"]["Get"][class_name])):
        search_result += results["data"]["Get"][class_name][i][properties[0]] + ".\n"
        for p in properties:
            print(results["data"]["Get"][class_name][i][p] + ".\n")

    return search_result


def analyze_research(path=""):

    '''
    questions = ["Based on the following excerpts from this research paper, return the list of the authors of this research paper",
                 "Based on the following excerpts from this research paper, return the list of the research methods used in this research paper.",
                 "Based on the following excerpts from this research paper, return the list of the key results presented in this research paper."]
    '''

    questions = ["Extract the author names based on the given context only. Do not make any assumptions.",
                 "Extract the main methods described in the given context only. Restrict the output to a list of 10 or less. Do not make any assumptions.",
                 "Extract the key results described in the given context only. Restrict the output to a list of 10 or less. Do not make any assumptions."] # "extract" works better than "return a list of"

    contexts = [get_context_for_authors(properties=["text"], k=3, path=path), get_context_for_methods(
        properties=["text"], k=3, path=path), get_context_for_key_results(properties=["text"], k=3, path=path)]
    
    for i in range(len(questions)):
        question = questions[i]
        context = contexts[i]
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Question: \"\"\"{question}\"\"\"\nContext: \"\"\"{context}\"\"\"\n",
            max_tokens=3000,
            temperature=0.2, # reduced the temperature
        )
        if i == 0:
            authors = response.choices[0].text
        elif i == 1:
            methods = response.choices[0].text
        elif i == 2:
            key_results = response.choices[0].text

    # make a json object with the following properties: authors, methods, key_results
    # return the json object
    return {"authors": authors, "methods": methods, "key_results": key_results}


async def upload_file(document_type, path, url, contentType="research"):
    # Load the file to Weaviate
    result = await load_pdf(class_name=class_name, properties={
                                     "type": contentType, "path": path, "url": url})

    contentType = "research" #TODO: get logic for different types of files (overwriting as a fix for now)
    print("uploaded file with path: ", path, " and content type: ", contentType)
    # if contentType is not "research" then we don't need to extract the authors, methods, and key results
    if contentType != "research":
        return

    analysis = analyze_research(path=path)
    print("analysis: ", analysis)
    
    # instead of returning the metadata, save it directly to the db
    SupabaseClient.update_metadata_in_database(path=path, metadata=analysis)
    print("updated metadata in database for path: ", path)
        
    return
