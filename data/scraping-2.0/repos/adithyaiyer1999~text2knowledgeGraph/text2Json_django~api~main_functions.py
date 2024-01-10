from . import openai_calls 
from . import prompts
from . import main_chunking_and_multithreading
from . import constants
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import json
import re

def is_url(string):
    regex = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain name
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, string) is not None

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from h1, h2, and other text
    tags = [
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'span', 'strong', 'em', 'blockquote',
    'ul', 'ol', 'li',
    'a',
    'img',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
    'form', 'input', 'textarea', 'button', 'select', 'option',
    'div', 'section', 'article', 'header', 'footer', 'nav', 'aside',
    'br', 'hr', 'label', 'iframe', 'script', 'style'
]
    text = ' '.join([tag.get_text() for tag in soup.find_all(tags)])
    return text

def get_youtube_id(url):
    regex = r"(?<=v=)[^&#]+"
    match = re.search(regex, url)
    return match.group() if match else None

def createGraphFromText_(text):
    # let's check if the text is only a url, if so parse url/youtube link
    if is_url(text):
        return createGraphFromUrl_(text)

    model =  "gpt-4-1106-preview"
    query_prompt = prompts.CREATE_HIGH_DETAIL_JSON_PROMPTS[model]
    prompt = query_prompt + "\n\n Text: " + text + "\n\nJSON:"
    response = openai_calls.ask_chatgpt(prompt, model)
    str_response = str(response)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")
    return str_response

def createGraphFromUrl_(url):
    if "youtube" in url:
        video_id = get_youtube_id(url)
        try:
            transcriptDict = YouTubeTranscriptApi.get_transcript(video_id)
            transcriptList = [a["text"] for a in transcriptDict]
            text = " ".join(transcriptList)
        except Exception as e:
            print(e)
            text = "This youtube video does not have a transcript readily available, please try with another link."
            return str({"Error : ": text})
    else:
        text = get_text_from_url(url)
    model = "gpt-4-1106-preview"
    query_prompt = prompts.CREATE_HIGH_DETAIL_JSON_PROMPTS[model]
    prompt = query_prompt + "\n\n Text: " + text + "\n\nJSON:"
    response = openai_calls.ask_chatgpt(prompt, model)
    str_response = str(response)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")
    return str_response

def addToGraphFromText_(text, json_text):
    model = "gpt-4-1106-preview"
    query_prompt = prompts.UPDATE_JSON_PROMPTS[model]
    prompt = query_prompt + "\n\n Paragraph : " + text + "\n\nJSON: " + json_text + " \n\nUpdated JSON:"
    response = openai_calls.ask_chatgpt(prompt, model)
    str_response = str(response)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")
    return str_response

def searchGraphFromText_(text, json_text):
    pathToNode, graph_dict = getMostRelevantNode(text, json_text)
    str_response = str(graph_dict)
    print("pathToNode: ", pathToNode)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")
    return str_response

def getMostRelevantNode(text, json_text):
    # lets store the entire path, and the graph which comes out of it
    pathToNode = []
    graph_dict = json.loads(json_text)
    original_graph_dict = graph_dict
    # pathToNode.append("Start")
    current_node = "Start"
    children_nodes = list(graph_dict.keys())
    IsCurrentNodeMostRelevant = "No"

    # Lets have 2 strikes, or 2 times that the model has to say "Yes", this is the most relevant node
    strikes = 0

    while IsCurrentNodeMostRelevant == "No":
        IsCurrentNodeMostRelevant, nextNode = getNextNodeFromOpenAI(current_node, children_nodes, text)
        print("IsCurrentNodeMostRelevant: ", IsCurrentNodeMostRelevant)
        print("nextNode: ", nextNode)
        current_node = nextNode
        pathToNode.append(current_node)
        if isinstance(graph_dict[current_node], str):
            print("reached leaf node")
            break
        children_nodes = list(graph_dict[current_node].keys())
        # if we have reached the right node, reply with a graph with this node as the root
        if IsCurrentNodeMostRelevant == "Yes":
            strikes = strikes+1 
            IsCurrentNodeMostRelevant="No"
        if strikes == 2:
            break
        graph_dict = graph_dict[current_node]
        if len(children_nodes) == 0:
            break
    
    # Now we try to create the entire path, and then the tree that comes out of it
    
    subtree = graph_dict
    for element in reversed(pathToNode):
        final_tree = {}
        final_tree[element] = subtree
        subtree = final_tree

    return pathToNode, final_tree

def answerQsFromTextAndGraph_(text, json_text):
    model =  "gpt-4-1106-preview"
    query_prompt = prompts.ANSWER_FROM_GRAPH_PROMPTS[model]
    prompt = query_prompt + "\n\n JSON : " + json_text + "\n\nQuestion: " + text + " \n\nAnswer:"
    response = openai_calls.ask_chatgpt(prompt, model)
    str_response = str(response)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")
    return str_response


def getNextNodeFromOpenAI(current_node,children_nodes, query):
    model = "gpt-4-1106-preview"
    query_prompt = prompts.TRAVERSE_GRAPH_PROMPTS[model]
    prompt = query_prompt + "\n\n Query: " + query + "\n\n Node: " + current_node + "\n\nChildren Nodes: " + sepearateListWithCommas(children_nodes) + "\n\Answer:"
    response = openai_calls.ask_chatgpt(prompt, model)
    str_response = str(response)
    
    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")

    json_output = json.loads(str_response)
    try:
        return json_output["IsCurrentNodeMostRelevant"], json_output["MostRelevantChildNode"]
    except Exception as e:
        print("Exception: ", e)
        return "Yes", "OPEN AI ERROR"

def sepearateListWithCommas(list):
    return ', '.join(list)

def createGraphFromTextIteratively_(text):

    model =  "gpt-4-1106-preview"
    query_prompt = prompts.CREATE_HIGH_DETAIL_JSON_PROMPTS[model]
    list_of_chunks = []
    # This function will create list of chunks, P.S. the chunks would be created based on max_token_length provided
    list_of_chunks = main_chunking_and_multithreading.chunk_text(text,constants.MAX_TOKEN_LENGTH)
    # This function will create list of JSON summaries, the function will call open ai api in multithreaded fashion
    list_of_json_summaries=main_chunking_and_multithreading.multithreaded_summarized_json(list_of_chunks,model,query_prompt)

    # Since our JSONs would have '\n', we need a different separator to identify jsons (list of json -> string)
    separator = "|||"

    # Combine JSON strings using the separator
    combined_json = separator.join(list_of_json_summaries)

    query_prompt = prompts.COMBINE_JSON_SUMMARIES[model]
    prompt = query_prompt + "\n\n LIST of JSON: " + combined_json + "\n\nMERGED JSON:"
    response = openai_calls.ask_chatgpt(prompt, model)

    str_response = str(response)

    # Some sanity text cleaning to avoid errors in yaml loading
    str_response = str_response.replace("json", "")
    str_response = str_response.replace("`", "")

#     print(list_of_json_summaries)
    return str_response





