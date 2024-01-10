
import json
import os
import sys
import boto3
#import utils
#import langchain
#from utils import bedrock, print_ww
#from langchain.llms.bedrock import Bedrock
#import requests
#from bs4 import BeautifulSoup
import re
#import newspaper3k
#from newspaper3k import Article
import base64
import io
#from PIL import Image
import warnings


def lambda_handler(event, context):  

    warnings.filterwarnings('ignore')

    module_path = ".."
    sys.path.append(os.path.abspath(module_path))

    # ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

    # os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
    # os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
    # os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."
    
    
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )
    
    inference_modifier = {
        "max_tokens_to_sample": 4000,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }
    
    textgen_llm = Bedrock(
        model_id="anthropic.claude-instant-v1",
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )
    
    print(boto3_bedrock)
    print(textgen_llm.client)
    
    # URL to scrape
    url = 'https://www.denver7.com/news/front-range/littleton'
    
    response = requests.get(url)
    
    # Ensure the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        exit()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all anchor tags with href attributes
    article_links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
    
        # Assuming that the article URLs will contain '/news/' (This might not always be the case)
        # You might need to adjust this condition based on the actual structure of the article links
        if '/news/' in href and href not in article_links:
            # Construct the full URL (if it's a relative link)
            if href.startswith('/'):
                href = 'https://www.denver7.com' + href
            article_links.append(href)
    
    # Join the URLs into a single string
    joined_urls = "\n".join(article_links)
    
    # Place the URLs into the desired format
    formatted_text = f"""
    We are aiming to curate a list of URLs to provide our users with specific news articles rather than general information or topic hubs. 
    To achieve this, we need to filter out URLs that don't meet certain criteria. Specifically, a valid news article URL will:
    
    1. Feature slugified article headlines, meaning the headlines will be represented in the URL with words separated by hyphens.
    2. This is extremely important: A valid url will contain more than four words in its slugified headline. For example: 'boulder-king-soopers-shooting' in the final part of the URL is not valid. 
    3. Not be too brief or composed of less than 120 characters in the headline.
    4. Not represent section fronts, topic hubs, or generic pages.
    
    
    Ensure that you consider the full context and topic of each URL, not just keywords in the headline. 
    Be thorough in evaluating all options, rather than prematurely stopping once you have found some matches. 
    Double check your output to ensure nothing is missing. 
    It is very important to provide our users with specific news stories. Sift through the provided list and select URLs that align with these requirements:
    
    {joined_urls}
    
    Assistant:"""
    
    response = textgen_llm(formatted_text)
    
    print_ww(response)
    
    inference_modifier = {
        "max_tokens_to_sample": 2048,
        "temperature": 0.125,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }
    
    textgen_llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )
    
    lines = response.strip().split('\n')
    
    article_urls = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('https://www.denver7.com'):
            # If the line ends with '-' and the next line is a URL, stitch them together
            if lines[i].endswith('-') and i < len(lines) - 1 and lines[i+1].startswith('https://www.denver7.com'):
                article_urls.append(lines[i] + lines[i+1].split('https://www.denver7.com')[-1])
                i += 2  # skip the next line since it's been stitched
            else:
                article_urls.append(lines[i])
                i += 1
        else:
            i += 1
    
    # Verify our URLs
    for url in article_urls:
        print(url)
        
    import re
    
    # User's topic of interest - for this example, let's say they're interested in "politics":
    user_topic = "good news"  # This will be provided by the user
    
    # Construct the refined prompt to find the top 5 articles that match the user's topic:
    refined_prompt = f"""
    \n\nHuman: Given the user's specific interest in the topic of '{user_topic}', we aim to find news articles that closely align with this topic. Considering the core subject and main theme of each article, not just specific keywords, please select the top 5 articles from the provided list that best match the user's interest. Do not change the urls in any of the articles, list them only. Here is the list:
    
    {article_urls}
    
    Assistant:"""
    
    # Now, you can feed this refined prompt to the Claude model to get the final set of URLs tailored to user preferences:
    refined_response = textgen_llm(refined_prompt)
    
    print_ww(refined_response)
    
    lines = refined_response.strip().split('\n')
    
    article_urls = []
    i = 0
    while i < len(lines):
        # Check if the line looks like the start of a URL
        if lines[i].startswith('https://www.denver7.com'):
            # If the line ends with '-', we assume it continues on the next line
            if lines[i].endswith('-') and (i < len(lines) - 1):
                stitched_url = lines[i] + lines[i+1].lstrip('https://www.denver7.com')
                article_urls.append(stitched_url)
                i += 2  # Move past the next line, since we've stitched it
            else:
                article_urls.append(lines[i])
                i += 1
        else:
            i += 1
    
    # Display the stitched URLs
    for url in article_urls:
        print(url)
        
    valid_urls = []
    
    for url in article_urls:
        try:
            response = requests.get(url, timeout=10)  # setting a timeout to ensure the request doesn't hang indefinitely
            if response.status_code != 404:  # not a "page not found" error
                valid_urls.append(url)
        except requests.RequestException:  # This will catch connection errors, timeouts, etc.
            print(f"Error while checking the URL: {url}")
    
    # Replace article_urls with the filtered list
    article_urls = valid_urls
    
    # If the list is empty, print an error message
    if not article_urls:
        print("Error: All articles lead to '404 page not found' errors or there were issues accessing them.")
    
    # Display the valid URLs
    for url in article_urls:
        print(url)
        
    summaries = []
    for article_url in article_urls:
        # Using newspaper3k to fetch and parse the article
        article = Article(article_url)
        article.download()
        article.parse()
        
        # Construct the prompt for summarization
        prompt = f"""
    \n\nHuman: Please provide a summary of the following text. Condense the summary to a single bullet point that captures the essence of the article while providing more information than simply the article title found in the url. It's crucial to offer the reader a succinct summary that effectively conveys the article's core message. This should be more detailed than the headline but concise enough for quick comprehension, ensuring the reader grasps the key points from news tailored to their specified keywords that they chose earlier. Keep this summary to a maximum of three sentences. Do not include any qualifying sentences before or after the bullet point summary. 
    {article.text}
    Assistant:"""
    
        # Construct the request body
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 4096,
            "temperature": 0.25,
            "top_k": 250,
            "top_p": 0.5,
            "stop_sequences": []
        })
    
        modelId = 'anthropic.claude-v2'  # Ensure you're using the desired model version
        accept = 'application/json'
        contentType = 'application/json'
    
        # Invoke the model to get the summary
        response_article = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body_article = json.loads(response_article.get('body').read())
    
        # Append the summary to our list
        summaries.append(response_body_article.get('completion'))
    
    # Print out the summaries
    for summary in summaries:
        print_ww(summary)
        
    module_path = ".."
    sys.path.append(os.path.abspath(module_path))
    
    # ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----
    
    # os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
    # os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
    # os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."
    
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None),
    )
    
    # Parameters for image generation
    negative_prompts = [
        "poorly rendered",
        "too detailed",
        "too realistic",
        "disfigured features",
        "not exciting", 
        "not interesting"
    ]
    style_preset='low-poly'
    modelId = "stability.stable-diffusion-xl"
    
    # Create data directory for storing images
    os.makedirs("data", exist_ok=True)
    
    # Iterate over summaries and generate images
    for idx, summary in enumerate(summaries):
        request = json.dumps({
            "text_prompts": (
                [{"text": summary, "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": 5,
            "seed": 5450 + idx,  # unique seed for each image
            "steps": 70,
            "style_preset": style_preset,
        })
    
        response = boto3_bedrock.invoke_model(body=request, modelId=modelId)
        response_body = json.loads(response.get("body").read())
    
        base_64_img_str = response_body["artifacts"][0].get("base64")
        image_path = f"data/image_{idx + 1}.png"
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
        image.save(image_path)
    
        print(f"Image for summary {idx + 1} saved at: {image_path}")
        
        return summaries
