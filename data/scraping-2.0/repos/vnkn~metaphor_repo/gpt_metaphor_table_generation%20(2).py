import html
from html.parser import HTMLParser
import pandas as pd
from metaphor_python import Metaphor
import openai
from dotenv import load_dotenv
import json
import concurrent.futures
import time
import os 
from flask import Flask, request, jsonify
import json
import concurrent.futures
import openai

#HTMLFilter is a custom parser to extract text content from HTML.
class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data
        
def finder(query: str) -> dict:
    """
    Uses the OpenAI API to suggest appropriate table attributes based on a provided query.

    :param query: A query string describing the topic.
    :return: A dictionary containing the list of suggested actions, the number of results that the model should return, and table columns.
    """
    functions = [
        {
            "name": "extract_data_from_query",
            "description": "Given a request, extract the given fields.",
            "parameters": {
                "type": "object",
                    "properties": {
                        "Todo": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "What are the actions required, in verb format?"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "How many results do we need to show? As a default, return 5 results, unless a specific number is specified, or the general feeling of the prompt is to return more or less results."
                        },
                        "relevant_field_list": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "Based on the topic and the actions, list the key subjects or attributes that would be most relevant to build a table about the topic. For example, if the topic was 'ML Engineers', relevant attributes might include 'Name', 'Education', and 'Location'."
                            },
                        }, 
            },
            "required": ["Todo", "relevant_field_list", "num_results"],
            }
        }
    ]
    
    
    # A list of 15 different examples used to provide context.
    messages = [
        {"role": "system", "content": "You are a digital assistant designed to understand prompts and suggest appropriate table attributes to organize the data related to the prompt."},
        {"role": "user", "content": "When I say something like: 'Find ML Engineers', I expect you to suggest relevant columns for a table. For this example, suitable columns might be 'Name', 'Website URL', Education', and 'Location'."},
        {"role": "system", "content": "Understood. Based on your prompt, I'll suggest relevant attributes or columns that would be most suitable for creating a table on that topic. Please provide the next query."},
        {"role": "user", "content": "When I say something like: 'Criticize these articles about Shakespeare', I expect you to suggest relevant columns for a table. Here the word criticize is especially relevant.  For this example, suitable columns might be 'Website URL','Summary', 'Issues', and 'Areas of Improvement'. Similarly, "},
        {"role": "system", "content": "Understood. Based on your prompt, I'll suggest relevant attributes or columns that would be most suitable for creating a table on that topic. Please provide the next query."},
        {"role": "user", "content": "When I ask for companies, I expect to you to suggest relevant columns based on what these companies do. For this example, suitable columns might be 'Name', 'Industry', 'Areas of Expertise', and 'Business Proposition'."},
        {"role": "system", "content": "Understood. Based on your prompt, I'll suggest relevant attributes or columns that would be most suitable for creating a table on that topic. Please provide the next query."},
        {"role": "user", "content": "Discover groundbreaking startups in the AI space."},
        {"role": "system", "content": "Relevant columns might be 'Startup Name', 'Founders', 'Funding Raised', 'Innovative Features', and 'Market Impact'."},

        {"role": "user", "content": "Unearth hidden gems in online education platforms."},
        {"role": "system", "content": "Relevant columns might be 'Platform Name', 'Courses Offered', 'User Reviews', 'Unique Selling Points', and 'Subscription Price'."},

        {"role": "user", "content": "Scour the web for captivating marketing campaigns from 2023."},
        {"role": "system", "content": "Relevant columns might be 'Campaign Name', 'Brand', 'Medium', 'Emotional Appeal', and 'Engagement Metrics'."},

        {"role": "user", "content": "Illuminate opportunities for aspiring writers."},
        {"role": "system", "content": "Relevant columns might be 'Opportunity Name', 'Platform', 'Remuneration', 'Submission Guidelines', and 'Expected Exposure'."},

        {"role": "user", "content": "Pursue top managerial talents in the retail sector."},
        {"role": "system", "content": "Relevant columns might be 'Name', 'Previous Company', 'Achievements', 'Leadership Style', and 'Contact Details'."},

        {"role": "user", "content": "Identify mesmerizing architectural wonders in Europe."},
        {"role": "system", "content": "Relevant columns might be 'Structure Name', 'Location', 'Architect', 'Historical Significance', and 'Visitor Reviews'."},

        {"role": "user", "content": "Dive into essential coding concepts for beginners."},
        {"role": "system", "content": "Relevant columns might be 'Concept Name', 'Description', 'Implementation Languages', 'Real-world Applications', and 'Learning Resources'."},

        {"role": "user", "content": "Unravel trends that are reshaping the financial sector."},
        {"role": "system", "content": "Relevant columns might be 'Trend Name', 'Financial Impact', 'Driving Forces', 'Future Predictions', and 'Companies Involved'."},

        {"role": "user", "content": "Spot thrilling opportunities in the world of e-sports."},
        {"role": "system", "content": "Relevant columns might be 'Event Name', 'Game', 'Participation Criteria', 'Prize Pool', and 'Sponsorship Opportunities'."},

        {"role": "user", "content": "Seek enchanting destinations for honeymooners."},
        {"role": "system", "content": "Relevant columns might be 'Destination', 'Country', 'Best Time to Visit', 'Romantic Activities', and 'Average Cost'."},

        {"role": "user", "content": "Probe into the most heartwarming philanthropic acts of 2023."},
        {"role": "system", "content": "Relevant columns might be 'Act Name', 'Individual/Organization', 'Beneficiaries', 'Funds Raised/Donated', and 'Media Coverage'."},

        {"role": "user", "content": "Unfold the journey of emerging tech influencers."},
        {"role": "system", "content": "Relevant columns might be 'Influencer Name', 'Specialization', 'Follower Count', 'Noteworthy Initiatives', and 'Collaboration Opportunities'."},

        {"role": "user", "content": "Delve into untapped markets for sustainable products."},
        {"role": "system", "content": "Relevant columns might be 'Market Name', 'Geographical Focus', 'Potential Size', 'Consumer Trends', and 'Major Players'."},

        {"role": "user", "content": "Explore exhilarating job roles in the world of cinema."},
        {"role": "system", "content": "Relevant columns might be 'Role Name', 'Description', 'Average Pay', 'Required Skills', and 'Career Growth'."},

        {"role": "user", "content": "Venture into the dazzling world of gemstone trade."},
        {"role": "system", "content": "Relevant columns might be 'Gemstone', 'Origin', 'Market Price', 'Trade Dynamics', and 'Investment Opportunities'."},

        {"role": "user", "content": "Engage with spellbinding arts from indigenous communities."},
        {"role": "system", "content": "Relevant columns might be 'Art Form', 'Community', 'Description', 'Cultural Significance', and 'Buying Options'."},

        {"role": "user", "content": "Channelize efforts towards combating climate change."},
        {"role": "system", "content": "Relevant columns might be 'Initiative Name', 'Global Impact', 'Key Players', 'Success Metrics', and 'Ways to Contribute'."},

        {"role": "user", "content": "Reach out to prodigious young talents in graphic design."},
        {"role": "system", "content": "Relevant columns might be 'Designer Name', 'Portfolio', 'Style', 'Clients Served', and 'Availability'."},

        {"role": "user", "content": "Relish the symphony of top classical compositions."},
        {"role": "system", "content": "Relevant columns might be 'Composition', 'Composer', 'Era', 'Instruments Used', and 'Popular Performances'."},

        {"role": "user", "content": "Navigate the vast seas of modern maritime innovations."},
        {"role": "system", "content": "Relevant columns might be 'Innovation', 'Purpose', 'Company', 'Economic Impact', and 'Adoption Rate'."},

        {"role": "user", "content": "Evaluate the top ML Startups."},
        {"role": "system", "content": "Relevant columns might be 'Company Name', 'Purpose', 'Technical Breadth', 'Funding', Strengths', and 'Weaknesses'."},

        {"role": "user", "content": 'Here is the query: ' + query}
        ]

    response = openai.ChatCompletion.create(
    #model="gpt-4-0613", 
    model="gpt-3.5-turbo-0613", 
    functions=functions,
    messages=messages,
    )
    try:
        args = response.choices[0].message.function_call.arguments
    except AttributeError:
        print("fallback")
        args = json.dumps({
            "Todo": ['Summarize'],
            "num_results": 5,
            "relevant_field_list": ['Article Name', 'Description']
        })
    return args

def infer_source_from_html(html_content: str, url: str) -> str:
    """
    Infer the main source or website name from a given URL and HTML content using OpenAI.

    :param html_content: The HTML content of the page.
    :param url: The URL of the page.
    :return: The inferred source or website name.
    """
    messages = [
        {"role": "user", "content": "Given the URL and HTML content, infer the display name or website name. Only use the html content if you can't deduce it from the URL."},
        {"role": "user", "content": f"URL: {url}\nHTML:\n{html_content}"},
        {"role": "system", "content": "Please extract and provide only the display name or main source (e.g. 'Wikipedia' for 'Wikipedia - The Free Encyclopedia'). Answer should be no more than a couple words." }
    ]
    # Call the OpenAI model to infer the source from the provided HTML and URL.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        
    )
    
    return response.choices[0].message.content.strip()


def extract_column_data(column, source_name, actions_request, prompt_content, max_tokens):
    """
    Extracts the data for a specific column from the content provided.
    
    :param column: The column for which the data needs to be extracted.
    :param source_name: The source inferred from the HTML content.
    :param actions_request: The actions to be performed formulated based on the verbs provided.
    :param prompt_content: The constructed prompt content to be sent for extraction.
    :return: The column and the extracted result or error message.
    """
    try:
        # Constructing message for data extraction
        messages = [{"role": "system", "content": f"{prompt_content}"}]
        
        # Sending request to OpenAI API and getting response
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages, max_tokens = max_tokens)
        
        # Returning the extracted result for the current column if available, else "Not Found"
        return column, response.choices[0].message.content.strip() if response.choices else "Not Found"
    except Exception as e:
        print(f"Error occurred while processing {column}: {e}")
        # Returning error message for the current column in case of an exception
        return column, "Error"

def summarize_with_memory_and_verbs_parallelized(html_content: str, verbs: list, url: str, columns: list, previous_summaries: list = None, max_tokens: int = None) -> dict:
    """
    Extracts data from HTML content in parallel for multiple columns based on the provided verbs and returns a dictionary containing the extracted results.
    
    :param html_content: The HTML content of the article.
    :param verbs: The list of verbs to formulate actions.
    :param url: The URL of the article.
    :param columns: The list of columns for which the data needs to be extracted.
    :param previous_summaries: List of summaries of the previous articles, default is None.
    :param max_tokens: The maximum number of tokens for the response, default is None.
    :return: A dictionary containing the extracted results for each column.
    """
    try:
        # Inferring the source from the HTML content
        source_name = infer_source_from_html(html_content, url)
        
        # Formulating action requests based on the verbs provided
        actions_request = ', '.join(verbs[:-1]) + ', and ' + verbs[-1]
        
        # Initializing result dictionary to store extracted results
        result_dict = {}
        
        # Constructing the prompt content for data extraction
        prompt_content = f"We have an article from {source_name}, and we have been asked to perform the following actions: {actions_request}.  Given this, please return the {{column}}, keeping it as succinct as possible(no more than 1 - 2 sentences). If possible, try to use different language from what was used in previous articles. Here are the summaries of the previous articles: {previous_summaries} and here is the html of the article: \n {html_content}."
        
        # Using ThreadPoolExecutor for parallel processing of columns
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submitting tasks for each column
            futures = [executor.submit(extract_column_data, column, source_name, actions_request, prompt_content.format(column=column), max_tokens) for column in columns]
            
            # Collecting results as tasks are completed
            for future in concurrent.futures.as_completed(futures):
                column, result = future.result()
                result_dict[column] = result
                
    except Exception as e:
        print(f"Error occurred: {e}")
        # In case of an exception, returning an error message for each column
        result_dict = {column: "Error" for column in columns}

    return result_dict

def summarize_with_memory_and_verbs(html_content: str, verbs: list, url: str, columns: list, previous_summaries: list = None, max_tokens: int = None) -> dict:
    """
    Extracts data from HTML content for multiple columns based on the provided verbs and returns a dictionary containing the extracted results.
    
    :param html_content: The HTML content of the article.
    :param verbs: The list of verbs to formulate actions.
    :param url: The URL of the article.
    :param columns: The list of columns for which the data needs to be extracted.
    :param previous_summaries: List of summaries of the previous articles, default is None.
    :param max_tokens: The maximum number of tokens for the response, default is None.
    :return: A dictionary containing the extracted results for each column.
    """
    try:
        # Inferring the source from the HTML content
        source_name = infer_source_from_html(html_content, url)
        
        # Formulating action requests based on the verbs provided
        actions_request = ', '.join(verbs[:-1]) + ', and ' + verbs[-1]
        
        # Initializing result dictionary to store extracted results
        result_dict = {}

        for column in columns:
            # Constructing the prompt content for data extraction
            prompt_content = f"We have an article from {source_name}, and we have been asked to perform the following actions: {actions_request}.  Given this, please return the {column}, keeping it as succinct as possible(no more than 1 - 2 sentences). If possible, try to use different language from what was used in previous articles. Here are the summaries of the previous articles: {previous_summaries} and here is the html of the article: \n {html_content}."
            
            # Constructing message for data extraction
            messages = [{"role": "system", "content": f"{prompt_content}"}]
            
            # Sending request to OpenAI API and getting response
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
            
            # Storing the extracted result for the current column if available, else "Not Found"
            result_dict[column] = response.choices[0].message.content.strip() if response.choices else "Not Found"
            
    except Exception as e:
        print(f"Error occurred: {e}")
        #

import concurrent.futures

def get_responses(client, example, columns=None, parallel=False):
    """
    Generate responses based on the given example and client object.

    :param client: The client object to interact with the backend service.
    :param example: The string input for which the responses need to be generated.
    :param columns: List of column names to be used; if not provided, defaults to None.
    :param parallel: Boolean flag to decide if the extraction should run in parallel, defaults to False.

    :return: A JSON string containing the response data or error messages.
    """

    # Initialize the response dictionary with an empty list for storing response items.
    response_dict = {"response": []}
    try:  # Outer try block to catch any unexpected errors.
        try:
            start_time = time.time()  # Track the start time of the operation.
            f = HTMLFilter()
            
            # Call client's search method with the example and store the results.
            results = client.search(example, use_autoprompt=True).results
            if not results:  # If no results are returned, update response dict with an error and return.
                response_dict["response"].append({'error': 'No results from client search', 'url': 'Unknown'})
                return json.dumps(response_dict)

            # Extract IDs from the results.
            ids = [result.id for result in results]

            # Call finder function with the example, replacing underscores with spaces.
            outputs = finder(example.replace("_", " "))
            if outputs is None:  # If finder function returns None, update response dict with an error and return.
                response_dict["response"].append({'error': 'finder function returned None', 'url': 'Unknown'})
                return json.dumps(response_dict)
            
            outputs = json.loads(outputs)  # Load the JSON string returned by finder function into a dictionary.
            verbs = outputs.get('Todo', [])  # Extract the 'Todo' field from outputs.

            # Call client's get_contents method with IDs and store the response.
            response = client.get_contents(ids)
            if not response.contents:  # If no contents are received, update response dict with an error and return.
                response_dict["response"].append({'error': 'No contents received from client.get_contents', 'url': 'Unknown'})
                return json.dumps(response_dict)
            
            # Decide on the columnlist to be used.
            columnlist = columns if columns else outputs.get('relevant_field_list', [])

            # Determine the number of results to be processed.
            num_results = min(outputs.get('num_results', 0), len(response.contents))
            if num_results == 0:  # If no results to process, update response dict with an error and return.
                response_dict["response"].append({'error': 'No results to process', 'url': 'Unknown'})
                return json.dumps(response_dict)

            # Prepare a list of tuples for parallel processing.
            q = [(content.url, f.feed(content.extract), f.text) for content in response.contents]

            # Define a function to extract data from each tuple.
            def extract_data(tup):
                url, _, s = tup
                # Decide on the extraction function based on the 'parallel' flag.
                extracted_data_func = summarize_with_memory_and_verbs_parallelized if parallel else summarize_with_memory_and_verbs
                try:
                    data, url = extracted_data_func(
                        s[0: 16385 - sum(len(s) for s in previous_summaries)],
                        verbs,
                        url,
                        columnlist,
                        previous_summaries,
                        max_tokens=50
                    ), url
                    data['url'] = url  # Update the 'url' field in the data dictionary.
                    return data, url  # Return the data dictionary and URL as a tuple.
                except Exception as e:
                    print(f"Error occurred during extraction: {e}")
                    return {"error": str(e), "url": url}, url  # Return error dictionary and URL as a tuple in case of an error.

            previous_summaries = []  # Initialize the list of previous summaries.

            # Execute the extraction function in parallel or sequentially based on the 'parallel' flag.
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(extract_data, item) for item in q[0:num_results]]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        data, url = future.result()  # Get the result of the future.
                        response_dict["response"].append(data)  # Append the data to the response dictionary.
                    except Exception as e:
                        print(f"Error occurred during future execution: {e}")
                        response_dict["response"].append({"error": str(e), "url": "Unknown"})  # Append error to the response dictionary in case of an error.
            
            # Add the 'key_columns' entry to the response dictionary.
            response_dict["key_columns"] = columnlist
            
            return json.dumps(response_dict)  # Convert the response dictionary to a JSON string and return.

        except Exception as e:  # Inner try block's except clause to catch errors in the main logic.
            print(f"An error occurred: {e}")
            response_dict["response"].append({"error": str(e), "url": "Unknown"})
            return json.dumps(response_dict)

    except Exception as e:  # Outer try block's except clause to catch any unexpected errors.
        print(f"An unexpected error occurred: {e}")
        response_dict["response"].append({"error": "An unexpected error occurred", "url": "Unknown"})
        return json.dumps(response_dict)

