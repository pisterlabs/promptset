import os
from atlassian import Confluence # See https://atlassian-python-api.readthedocs.io/index.html
import nltk
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
import openai
import numpy as np

# See the companion Medium article https://medium.com/@francois.ascani/running-chatgpt-on-your-internal-confluence-documentation-d7761aa8fc68
# for a detailed description of the code

# Set the API key
openai.api_key = os.environ["openai-api-key"]

def get_url():
    return 'https://my-url.atlassian.net'

def connect_to_Confluence():
    '''
    Connect to Confluence
    
    We use the API token for the cloud
    To create an API token here: Confluence -> Profile Pic -> Settings -> Password -> Create and manage tokens
    
    Return
    ------
    A connector to Confluence
    '''
    
    url = get_url()
    username = os.environ["jira-username"]
    password = os.environ["jira-api-token"]
    
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        cloud=True)
    
    return confluence
  
def get_all_pages(confluence, space='MY-SPACE'):
    '''
    Get all the pages within the MY-SPACE space.
    
    Parameters
    ----------
    confluence: a connector to Confluence
    space: Space of the Confluence (i.e. 'MY-SPACE')
    
    Return
    ------
    List of page objects. Each page object has all the information concerning
    a Confluence page (title, body, etc)
    '''
    
    # There is a limit of how many pages we can retrieve one at a time
    # so we retrieve 100 at a time and loop until we know we retrieved all of
    # them.
    keep_going = True
    start = 0
    limit = 100
    pages = []
    while keep_going:
        results = confluence.get_all_pages_from_space(space, start=start, limit=100, status=None, expand='body.storage', content_type='page')
        pages.extend(results)
        if len(results) < limit:
            keep_going = False
        else:
            start = start + limit
    return pages
  
def get_doc_model():
    '''
    Model string to calculate the embeddings.
    '''
    return 'text-embedding-ada-002'

def get_embeddings(text: str, model: str) -> list[float]:
    '''
    Calculate embeddings.

    Parameters
    ----------
    text : str
        Text to calculate the embeddings for.
    model : str
        String of the model used to calculate the embeddings.

    Returns
    -------
    list[float]
        List of the embeddings
    '''
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_max_num_tokens():
    return 2046

def collect_title_body_embeddings(pages, save_csv=True):
    '''
    From a list of page objects, get the title and the body, calculate
    the number of tokens as well as the embeddings of the body.
    
    Parameters
    ----------
    pages: List of page objects, i.e. output of get_all_pages()
    save_csv: Boolean. If True, the dataframe is saved locally
    into a CSV file.
    
    Return
    ------
    A dataframe of the title and body of all pages.
    '''

    collect = []
    for page in pages:
        title = page['title']
        link = get_url() + '/wiki/spaces/MY-SPACE/pages/' + page['id']
        htmlbody = page['body']['storage']['value']
        htmlParse = BeautifulSoup(htmlbody, 'html.parser')
        body = []
        for para in htmlParse.find_all("p"):
            # Keep only a sentence if there is a subject and a verb
            # Otherwise, we assume the sentence does not contain enough useful information
            # to be included in the context for openai
            sentence = para.get_text()
            tokens = nltk.tokenize.word_tokenize(sentence)
            token_tags = nltk.pos_tag(tokens)
            tags = [x[1] for x in token_tags]
            if any([x[:2] == 'VB' for x in tags]): # There is at least one verb
                if any([x[:2] == 'NN' for x in tags]): # There is at least noun
                    body.append(sentence)
        body = '. '.join(body)
        # Calculate number of tokens
        tokens = tokenizer.encode(body)
        collect += [(title, link, body, len(tokens))]
    DOC_title_content_embeddings = pd.DataFrame(collect, columns=['title', 'link', 'body', 'num_tokens'])
    # Caculate the embeddings
    # Limit first to pages with less than 2046 tokens
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
    doc_model = get_doc_model()
    DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, doc_model))
    
    if save_csv:
        DOC_title_content_embeddings.to_csv('DOC_title_content_embeddings.csv', index=False)
        
    return DOC_title_content_embeddings
  
def update_internal_doc_embeddings():
    # Connect to Confluence
    confluence = connect_to_Confluence()
    # Get page contents
    pages = get_all_pages(confluence, space='MY-SPACE')
    # Extract title, body and number of tokens
    DOC_title_content_embeddings= collect_title_body_embeddings(pages, save_csv=True)
    return DOC_title_content_embeddings
  
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))
  
def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_model = get_doc_model()
    query_embedding = get_embeddings(query, model=query_model)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)
    
    return doc_embeddings
  
def construct_prompt(query, doc_embeddings):
    
    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_links = []
     
    for section_index in range(len(doc_embeddings)):
        # Add contexts until we run out of space.        
        document_section = doc_embeddings.loc[section_index]
        
        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
        chosen_sections_links.append(document_section.link)
        
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"
    
    return (prompt,  chosen_sections_links)
  
  def internal_doc_chatbot_answer(query, DOC_title_content_embeddings):
    
    # Order docs by similarity of the embeddings with the query
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)
    # Construct the prompt
    prompt, links = construct_prompt(query, DOC_title_content_embeddings)
    # Ask the question with the context to ChatGPT
    COMPLETIONS_MODEL = "text-davinci-002"

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    output = response["choices"][0]["text"].strip(" \n")
    
    return output, links
  
