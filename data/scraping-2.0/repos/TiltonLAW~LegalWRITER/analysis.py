import os
import json
import openai  

# analysis.py

def is_narrative(parenthetical):
    return len(parenthetical.split()) > 5

def check_relevancy_with_text_ada(parenthetical, prompt, query):
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Completion.create(
        engine="text-ada-001",
        prompt=f"Does following statement: '{parenthetical}' directly answer the following question: '{prompt}'? Answer only 'Yes' or 'No'. Your answer should contain no other words.",
        max_tokens=7,
        temperature=0.2,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].text.strip() == 'Yes'

def parse_json(json_file):
    narrative_parentheticals = []
    
    with open(json_file) as file:
        data = json.load(file)
        for citation in data.get('cites_to', []):
            for pin_cite in citation.get('pin_cites', []):
                parenthetical = pin_cite.get('parenthetical', '')
                if is_narrative(parenthetical):
                    narrative_parentheticals.append(parenthetical)
                    
    return narrative_parentheticals

def filter_and_rank_cases(prompt, query, search_results):
    case_infos = []
    research_body = []
    query = search_results['query']  # Extract keywords from the query

    if 'results' not in search_results:
        return research_body, case_infos

    results = search_results['results']  

    # Process each case
    for case in results:
        # print(f"Processing case: {case['name']}\n\n")  # Debug print

        cites_to = case.get('cites_to', [])
        case_parentheticals = []
        for cite in cites_to:
            pin_cites = cite.get('pin_cites', [])
            # Filter out parentheticals that aren't narrative
            parentheticals = [pin_cite.get('parenthetical') for pin_cite in pin_cites if pin_cite.get('parenthetical') and is_narrative(pin_cite.get('parenthetical'))]
            # Filter out irrelevant parentheticals
            relevant_parentheticals = [parenthetical for parenthetical in parentheticals if check_relevancy_with_text_ada(parenthetical, prompt, query)]
            if relevant_parentheticals:
                print(f"Relevant parentheticals : {relevant_parentheticals}\n\n")  # Debug print
                case_parentheticals.extend(relevant_parentheticals)

        # Get the official citation
        citations = case.get('citations', [])
        official_cite = next((c['cite'] for c in citations if c['type'] == 'official'), '')

        # Format the citation as 'Name, Citation (Year)'
        formatted_citation = f"{case['name_abbreviation']}, {official_cite} ({case['decision_date'][:4]})" if official_cite else ''

        # Add the slim case text to the research_body
        slim_case_text = {
            "name_abbreviation": case['name_abbreviation'],  
            "citation": formatted_citation, 
            "parentheticals": case_parentheticals,
            "html_url": case['frontend_url'],
            "url": case['url']  # Add the JSON case URL
        }   
        research_body.append(slim_case_text)

    # Iterate over the research body
    for slim_case_text in research_body:
        # Create a case information dictionary
        case_info = {
            "name_abbreviation": slim_case_text['name_abbreviation'],
            "citation": slim_case_text['citation'],
            "helpful_parenthetical": ', '.join(slim_case_text['parentheticals']),
            "html_url": slim_case_text['html_url'],
            "url": slim_case_text['url']  # Include the JSON case URL
        }
    
        # Add the case information dictionary to the list
        case_infos.append(case_info)

    return research_body, case_infos