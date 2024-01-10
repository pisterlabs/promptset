from server.embeddings import openai
import concurrent.futures

from openai.embeddings_utils import distances_from_embeddings


def get_related_info(keyword,context):
    """
    Extract related information from the context
    """
    response = openai.ChatCompletion.create(  # Create a completions using the keyword and context
            messages = [{
                "role":"user",
                "content": f"""
                    Extract information most related to {keyword} of the following context which was taken from a research paper\n
                    context : {context}\n
                    points :
                    """
            }],
            temperature=0.5,
            max_tokens = 1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model="gpt-3.5-turbo",
        )
    return response["choices"][0]["message"]["content"].strip()

def generate_content(
    context,
    keyword,
    model="gpt-3.5-turbo",
    stop_sequence=None
):
    """
    Generate content based on the generated points of the paper
    """
    response = openai.ChatCompletion.create(  # Create a completions using the keyword and context
            messages = [{
                "role":"user",
                "content": f"""
                Organize the following points related to {keyword} of a research by dividing into suitable subtopics. Generate a summerized paragraph for each subtopic\n\n
                use this format,\n
                ## generated subtopic ##\n
                <Summarized paragraph under the subtopic>\n\n
                points: {context} 
                organized document:   
                 """
            }],
            temperature=0.5,
            max_tokens = 2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
    return response["choices"][0]["message"]["content"].strip()
  


def content_dict(txt):
    """
    Create a dictionary of generated content by sections
    """
    txt = txt.replace('\n','')
    sections = txt.split('##') # split the text into topics and content using the '##' delimiter
    topics = sections[1::2]
    content = sections[2::2]

    dict = [{'Title': topics[i], 'Content': content[i]} for i in range(len(topics))] # Create a list of dictionaries using a list comprehension

    return dict

def match_diagrams(diagrams_df,generated_content_dict,threshold = 0.15):
    """
    match diagrams for each generated section
    """
    for section in generated_content_dict:
        content_embeddings = openai.Embedding.create(input=section['Content'], engine='text-embedding-ada-002')['data'][0]['embedding'] #Get Embeddings
        diagrams_df['Distances'] = distances_from_embeddings(content_embeddings, diagrams_df['Embeddings'].values, distance_metric='cosine')  # Get the distances from the embeddings

        diagrams_df = diagrams_df.sort_values('Distances', ascending=True) # sort ascending as distances
        
        if diagrams_df['Distances'][0] < threshold:
            section['Diagrams'] = {'Type':diagrams_df['Type'][0],'Figure':diagrams_df['Figure'][0],'Description':diagrams_df['Text'][0]}
            # diagrams_df = diagrams_df.drop(index = 0)

    return generated_content_dict


def Generate(content_df,diagrams_df,keyword):
    """
    Main function for generating
    """
    information = [None] * len(content_df)  # Initialize the list with None values
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, row in content_df.iterrows():
            context = row["Text"]
            keyword = keyword
            futures[executor.submit(get_related_info, context, keyword)] = i  # Use a dictionary to associate each future with an index
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]  # Get the index associated with the completed future
            information[i] = future.result()  # Add the result to the appropriate index in the list

    related_information = ("\n").join(information)

    generated_content = generate_content(related_information,keyword)

    generated_content_dict = content_dict(generated_content) # create dictionary

    generated_content_dict = match_diagrams(diagrams_df,generated_content_dict) # match diagrams

    return generated_content_dict