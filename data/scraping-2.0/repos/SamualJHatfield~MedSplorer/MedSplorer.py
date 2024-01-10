import openai
import json
from Bio import Entrez, Medline
from collections import defaultdict
from collections import Counter
import requests
from xml.etree import ElementTree
from lxml import etree
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Tk, Text, Entry, Button, Label, OptionMenu, StringVar
from ttkthemes import ThemedTk


n = 10
all_references = []
#Extracts Keywords from user query, gpt query, or literature review category
def extract_keywords(query):
    prompt = f"Given the following query, identify the ideal PubMed Mesh sequence to search:\n\n{query} .Present only mesh terms in the proper format with no additional numbering, labeling, or explanation"
    keywords = evaluate_gpt4(prompt)
    return keywords

# PubMed search function
def search_pubmed(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]
    handle2 = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle2)
    results = [{"id": id, "record": record} for id, record in zip(id_list, records)]
    handle2.close()

    # Retrieve reference data for each paper
    for result in results:
        id = result["id"]
        handle3 = Entrez.elink(dbfrom="pubmed", id=id, cmd="neighbor_score")
        record3 = Entrez.read(handle3)
        handle3.close()

        references = []
        if record3 and "LinkSetDb" in record3[0]:
            for link in record3[0]["LinkSetDb"]:
                if link["LinkName"] == "pubmed_pubmed_refs":
                    references = [ref["Id"] for ref in link["Link"]]
                    break
        result["record"]["CR"] = references

    return results

#call GPT
def evaluate_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a medical research assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    print(response)  # Debug print to see the response
    return response['choices'][0]['message']['content'].strip()

def science_questions(query, search_type="all"):
    text = []
    category_references = []
    max_results = n
    while True:
        # Search PubMed for the user's query
        if search_type == "review":
            pubmed_results = search_pubmed(extract_keywords(query) + " AND review[ptyp]", max_results)
        else:
            pubmed_results = search_pubmed(extract_keywords(query), max_results)

        # Extract titles and abstracts for the first n papers
        if len(pubmed_results) < 1:
            text.append(f"No related results found in PubMed for query: {query}")
            category_references.append("No References in pubmed")
            break
        else:
            top_papers = []
            category_papers = []
            for result in pubmed_results:
                title = result["record"].get("TI", "")
                abstract = result["record"].get("AB", "")
                author = result["record"].get("FAU", "")
                year = result["record"].get("DP", "")
                Pubid = result["record"].get("PMID", "")
                references = result["record"].get("CR", "")
                top_papers.append({"id": Pubid, "title": title, "abstract": abstract, "author": author, "year": year})
                    
            # Send titles and abstracts to OpenAI API as one query
            papers_text = "\n\n".join([f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nAuthor: {paper['author']}\nYear: {paper['year']}" for paper in top_papers])
            prompt = f" Pubmed results: {papers_text} \n\n Instructions: Using the provided PubMed search results, identify a comprehensive list of specific questions to ask or hypotheses to test that may not yet have been explored in the field of {query}"
            #Checks if prompt is too long and calls GPT to generate questions
            if len(prompt) / 4 < 3597:
                Pubmed_result = evaluate_gpt4(prompt)
                text.append(Pubmed_result)
                for result in top_papers:
                    #catelogs a formatted version of the category references, adding them to both the total references as well as the category references
                    # Check if 'author' is a list and not empty before accessing its first element
		    author = result["author"][0] if result["author"] and isinstance(result["author"], list) else "Author not available"

                    all_references.append([author, result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                    category_papers.append([result["author"][0],  result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                #includes references used specifically in this query
                category_references.append(category_papers)
                break
            else:
                max_results -= 1
                if max_results <= 0:
                    questions.append(f"No related results found in PubMed for query: {query}")
                    category_references.append("No References in pubmed")
    return text, category_references, all_references
                
def literature_review(user_query, search_type="all"):
    text = []
    category_references = []
    max_results = n
    while True:
        if search_type == "review":
            pubmed_results = search_pubmed(extract_keywords(user_query) + " AND review[ptyp]", max_results)
        else:
            pubmed_results = search_pubmed(extract_keywords(user_query), max_results)

        # Extract titles and abstracts for the first n papers
        if len(pubmed_results) < 1:
            text.append(f"No related results found in PubMed for query: {user_query}")
            category_references.append("No References in pubmed")
            break
        else:
            top_papers = []
            category_papers = []
            for result in pubmed_results:
                title = result["record"].get("TI", "")
                abstract = result["record"].get("AB", "")
                author = result["record"].get("FAU", "")
                year = result["record"].get("DP", "")
                Pubid = result["record"].get("PMID", "")
                references = result["record"].get("CR", "")
                top_papers.append({"id": Pubid, "title": title, "abstract": abstract, "author": author, "year": year})
                    
            # Send titles and abstracts to OpenAI API as one query
            papers_text = "\n\n".join([f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nAuthor: {paper['author']}\nYear: {paper['year']}" for paper in top_papers])
            prompt = f" Pubmed results: {papers_text} \n\n Instructions: Using the provided PubMed search results, write a comprehensive reply to the given topic. Make sure to provide in-text citations using [author, year] notation after the reference. Do not include a list of references. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. \n\n Topic: {user_query}"
            #Checks if prompt is too long and calls GPT to generate questions
            if len(prompt) / 4 < 3597:
                Pubmed_result = evaluate_gpt4(prompt)
                text.append(Pubmed_result)
                for result in top_papers:
                    #catelogs a formatted version of the category references, adding them to both the total references as well as the category references
                    all_references.append([result["author"][0],  result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                    category_papers.append([result["author"][0],  result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                #includes references used specifically in this query
                category_references.append(category_papers)
                break
            else:
                max_results -= 1
                if max_results <= 0:
                    questions.append(f"No related results found in PubMed for query: {user_query}")
                    category_references.append("No References in pubmed")
    return text, category_references, all_references

def custom_prompt(user_query, search_type="all", custom_instructions="", n=10):
    text = []
    category_references = []
    instructions ={custom_instructions}
    max_results = n
    while True:
        if search_type == "review":
            pubmed_results = search_pubmed(extract_keywords(user_query) + " AND review[ptyp]", max_results)
        else:
            pubmed_results = search_pubmed(extract_keywords(user_query), max_results)

        # Extract titles and abstracts for the first n papers
        if len(pubmed_results) < 1:
            text.append(f"No related results found in PubMed for query: {user_query}")
            category_references.append("No References in pubmed")
            break
        else:
            top_papers = []
            category_papers = []
            for result in pubmed_results:
                title = result["record"].get("TI", "")
                abstract = result["record"].get("AB", "")
                author = result["record"].get("FAU", "")
                year = result["record"].get("DP", "")
                Pubid = result["record"].get("PMID", "")
                references = result["record"].get("CR", "")
                top_papers.append({"id": Pubid, "title": title, "abstract": abstract, "author": author, "year": year})
                    
            # Send titles and abstracts to OpenAI API as one query
            papers_text = "\n\n".join([f"Title: {paper['title']}\nAbstract: {paper['abstract']}\nAuthor: {paper['author']}\nYear: {paper['year']}" for paper in top_papers])
            prompt = f" Pubmed results: {papers_text} \n\n Using the provided PubMed search results, follow the given instructions: {instructions} Do not include a list of references. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. \n\n Topic: {user_query}"
            #Checks if prompt is too long and calls GPT to generate questions
            if len(prompt) / 4 < 3597:
                Pubmed_result = evaluate_gpt4(prompt)
                text.append(Pubmed_result)
                for result in top_papers:
                    #catelogs a formatted version of the category references, adding them to both the total references as well as the category references
                    all_references.append([result["author"][0],  result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                    category_papers.append([result["author"][0],  result["year"], result["title"], f" https://pubmed.ncbi.nlm.nih.gov/{result['id']}/"])
                #includes references used specifically in this query
                category_references.append(category_papers)
                break
            else:
                max_results -= 1
                if max_results <= 0:
                    text.append(f"No related results found in PubMed for query: {user_query}")
                    category_references.append("No References in pubmed")
    return text, category_references, all_references


def export_bibtex(references, output_file_path):
    with open(output_file_path, "w") as f:
        for ref in references:
            author, year, title, url = ref
            pmid = url.split("/")[-2]
            bibtex_entry = f'''@article{{{pmid},
                author = {{{author}}},
                title = {{{title}}},
                year = {{{year}}},
                url = {{{url}}},
                note = {{PMID: {pmid}}}
            }}\n'''
            f.write(bibtex_entry)

            
def submit_query():
    global operation_type  # Add this line to access the global variable
    
    Entrez.email = email_entry.get()  # Get the email from the entry field
    openai.api_key = api_key_entry.get()  # Get the API key from the entry field
    query = query_entry.get()
    search_type = search_type_var.get()
    operation = operation_type.get()  # Assign the result of get() method to a new variable
    
    if operation == "literature review":
        text, category_references, all_references = literature_review(query, search_type)
        result_text = "\n\n".join(text)
    elif operation == "relevant questions":
        text, category_references, all_references = science_questions(query, search_type)
        result_text = "\n\n".join(text)
    elif operation == "custom prompt":
        custom_instructions = custom_instructions_entry.get()  # Get the custom instructions from the entry field
        n = 10  # You can change this value depending on the number of results you want to fetch
        text, category_references, all_references = custom_prompt(query, search_type, custom_instructions, n)
        result_text = "\n\n".join(text)
    
    # Format category_references for display
    result_category_references = "\n\n".join([f"{idx + 1}. {', '.join(ref)}" for idx, ref in enumerate(category_references[-1])])

    result_textbox.delete(1.0, tk.END)
    result_textbox.insert(tk.END, f"{query}:\n\n{result_text}\n\nCategory References:\n\n{result_category_references}\n\n")

def copy_to_clipboard():
    result_text = result_textbox.get(1.0, tk.END)
    root.clipboard_clear()
    root.clipboard_append(result_text)

def export_to_zotero():
    file_path = filedialog.asksaveasfilename(defaultextension=".bib", filetypes=[("BibTeX files", "*.bib"), ("All files", "*.*")])
    if file_path:
        export_bibtex(all_references, file_path)

def update_custom_instructions_visibility(*args):
    if operation_type.get() == "custom prompt":
        custom_instructions_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        custom_instructions_entry.grid(row=5, column=1, padx=(0, 10))
    else:
        custom_instructions_label.grid_remove()
        custom_instructions_entry.grid_remove()


root = ThemedTk(theme="blue")
root.title("MedSplorer")

style = ttk.Style()
style.configure("TButton", font=("Sans-serif", 12))
style.configure("TLabel", font=("Sans-serif", 12))
style.configure("TEntry", font=("Sans-serif", 12))
style.configure("TCombobox", font=("Sans-serif", 12))
style.configure("TText", font=("Sans-serif", 12))

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

email_label = ttk.Label(frame, text="Email:")
email_label.grid(row=0, column=0, sticky=tk.W, pady=5)
email_entry = ttk.Entry(frame, width=50)
email_entry.grid(row=0, column=1, pady=5)

api_key_label = ttk.Label(frame, text="API Key:")
api_key_label.grid(row=1, column=0, sticky=tk.W, pady=5)
api_key_entry = ttk.Entry(frame, width=50)
api_key_entry.grid(row=1, column=1, pady=5)

query_label = ttk.Label(frame, text="Query:")
query_label.grid(row=2, column=0, sticky=tk.W, pady=5)
query_entry = ttk.Entry(frame, width=50)
query_entry.grid(row=2, column=1, pady=5)

search_type_label = ttk.Label(frame, text="Search type:")
search_type_label.grid(row=3, column=0, sticky=tk.W, pady=5)
search_type_var = tk.StringVar()
search_type_combobox = ttk.Combobox(frame, textvariable=search_type_var, values=("all", "review"), state="readonly", width=47)
search_type_combobox.set("all")
search_type_combobox.grid(row=3, column=1, pady=5)

operation_type_label = ttk.Label(frame, text="Operation type:")
operation_type_label.grid(row=4, column=0, sticky=tk.W, pady=5)
operation_type = StringVar(root)
operation_type.set("literature review") # default value
operation_options = OptionMenu(frame, operation_type, "literature review", "relevant questions", "custom prompt")
operation_options.grid(row=4, column=1)
operation_type.trace("w", update_custom_instructions_visibility)

custom_instructions_label = ttk.Label(frame, text="Custom instructions:")
custom_instructions_label.grid(row=5, column=0, sticky=tk.W, pady=5)
custom_instructions_entry = ttk.Entry(frame, width=50)
custom_instructions_entry.grid(row=5, column=1, padx=(0, 10))
update_custom_instructions_visibility()

submit_button = ttk.Button(frame, text="Submit", command=submit_query)
submit_button.grid(row=6, column=0, sticky=tk.E, pady=10)

result_textbox = tk.Text(root, wrap=tk.WORD, width=80, height=20, yscrollcommand=None)
result_textbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
result_textbox.insert(tk.END, '''If you like a response, copy it to your clipboard before your next query!\n\nInstructions:\n\n1) Enter a valid email address in the 'EMail' field. This is a requirement of the PubMed API. \n\n2) Enter your OpenAI API key in the 'OpenAI' field. to generate an API Key (write this down somewhere!), navigate to https://platform.openai.com/account/api-keys, log in, and select 'Create API Key.' I don't want to charge for the use of this app, so you pay as you go! Each query costs a little less than one cent.\n\n3)Input your query! The app will extract keywords and search pubmed for the most relevant papers, where the titles and abstracts will be fed through the GPT-3.5 turbo API.\n\n4)The 'Search Type' feature lets you specify if you want to search just review papers or all papers available.\n\n5)The 'Operation Type' is where the magic happens! Selecting 'Literature Review' will prompt the app to create a summary of the information it finds. 'Relevant Questions' will pose some potential questions you can explore in the literature to further hone what you want to investigate. 'Custom Prompt' lets you format the response to your pubmed search in any manner! An example would be to prompt it to create practice questions, provide an example outline for a review paper, etc.\n\n6)'Submit' initiates the search. Because the app is bouncing between multiple APIs to maximize the number of search results and provide the best answer, it should take between 20 and 40 seconds to process each request. Your query, an answer to your query with in-text citations, and a list of all references will generate in the main box.\n\n7) After each query, if you like the response then 'Copy to Clipboard' allows you to temporarily copy the entire generated response to paste into whatever word processor of your choice.\n\n8) At the end of your entire session, if you would like to save all of the references you have generated, click on "save to Zotero" to save a .bib file which can be imported to the Citation management system. For more information on Zotero, see here: https://www.zotero.org/''')

scrollbar = ttk.Scrollbar(root, orient="vertical", command=result_textbox.yview)
scrollbar.grid(row=1, column=1, sticky="ns")

result_textbox["yscrollcommand"] = scrollbar.set

copy_button = ttk.Button(frame, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.grid(row=7, column=3, sticky=tk.E, pady=10)

export_button = ttk.Button(frame, text="Export results to Zotero", command=export_to_zotero)
export_button.grid(row=6, column=3, sticky=tk.E, pady=10)

root.mainloop()