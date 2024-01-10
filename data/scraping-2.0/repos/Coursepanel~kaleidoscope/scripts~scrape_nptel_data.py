import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import htmlmin
import openai
import json
from concurrent.futures import ProcessPoolExecutor
import os
import csv
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

from builtins import str

auth_token="g_a=820k7bwIBtvWUFPzbkvtnHXZL8tG6nn6wKGtC7dG8K88RrO4AxIyLh8XuG18v9wa; g_b=poC8ll+QvTXgYYHFuF9liOQpdLn/ji89TDzlfI/xfuC7QrLLh8tIgnckL1RGrxTLfOImCqIoyQxm2ewccnFl/6LdJf9ftY9RxdyHCq/iJFw=; g_c=vhrba5TuVTzoMqPY/6ev0EM/3w36y7kv+uI9EjjODuN96CH45Ua8Hs2mwVnvY94T/ou8eDg3PIRrrs0juToSGOecYwCZ0o1oM2V9AA10r9nH+fuRtZ8bm8FV2gvsPnKPooHtJlPGiXe/wgff46ddNLM025uR9wXFPSe/s0EwF0xblKdY0ExzpCX+oomFk2uEYIcyMRbE2iZg5GkuDqscFtY5iLEnqIhOiNQ1PJCaoyfiF3GH4K0BIlo/6zu1kXza0uX6jmORw36esiX6UGjNNvlnCSNNSt1nuf281AmG8/gx/k4Dx9GjICn01pQ+38umRtq5RJRWsLCO98PSHc+eXF/bIE4IG8GhBpPdizZK9/Q/JiMagbTTfBsZsfULygxTYHBvv//OPtArdjkA+dfMS95TjM2SFmsY/OajxIDrHMAdg8YG9GKa7lqSAWAYKD0PvmPF4lQDEMO5TM1tSG859qDH19wrd3NHAgpCmJggBNmu3eTeS0t53t1PT4CB5W6qEg9A+bCVOOrScCibHYLN39xGXuAwGVxO+8JIL27XUYe6EBhCjInlvEdZ5qYDkF9ZDlSRDDz5bC/Dekx5pHc3ay3rlcS88nYkaoWjMyL1NJn1biHkNxjqQ3da40hmWtXe8b3HSufJjJLITE1ANDy3e0+7Mc+YBYcBAmPvkFF1tjTiP9St6RJ0Mn2Iwx3VKSwxNQoPgbIYhHKDrcEpzBZGGZsVDrVgg/Y5kITEj8/TlUbzfhIcdx4FnbrmpvF86KWdGMKUmuN/+iXbMH/Ho1xCnMVRMAf0pn55vfKxCPV39ax1V6/2Eb87dMCXNwCLVwF33wCvuhldGPlGBhS3Jqvq/XiN1v2A3c66GSHJVdhkb1BML8Z/PeBKD0NQ+zF/biOw4XKDNRUcXjEg6vPh9djqdjVbz1iO/H6PQyLt5E/Lrd08/c7wEhTs787MAqHWYOjfimERb4jwH5xmkPYUduudJWK/XWMc1YhqukbU40qzUHrp0onCtYAoVOKN1fCIdWOk84XnVIUIn4RmIiH5bRU887nTIaCU/mxBH57sxiLaPvGJH73w+snMyJ0Tode9AU6u2jUOsa31/KSDNh2r3Qo6QDgs48wepVi3E1JrwN1r2F7jd6zRb27ItNmPK7lNXq0zOTIemEU+T/glOERhV/vu7ppDzg+nwd24TE99Ct4h4WvQZPA/6sAIqkFlZ/cdLus1Yxnm64INHN0R6suJQHx8+BVIQvOAbIeFZjUQZYDmXn5mdkMOJ8sRYQnBWQM0lK3Ge4nbYqVU1yPHoo4+bq0XNxO7YeadKDlievMSJuPYlVdG0N4zJSkYlS/lnXLjjQOy; amp_adc4c4=TZzwzz4je6WzlnQ9OHCTUJ...1gvqmhjff.1gvqmhkkr.0.0.0"
headers = {"Authorization": f"Bearer {auth_token}"}

def find_text(soup, keyword):
    keyword_lower = keyword.lower()
    element = soup.find(lambda tag: tag.name ==
                        'span' and keyword_lower in tag.text.lower())
    return element.text if element else ''


# Function to scrape data from course page
def scrape_data(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print('=> 2.Received HTML data for url.... ⏰')
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

    # with open('response.txt', 'w') as file:
    #     file.write(response.text)

    print('=> 3.Cleaning and processing HTML using beautiful soup 🍜')
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove all unecessary tags
    for tag in soup(['script', 'style','link','head','img','button','option','a','br','footer','section','header','table','tr','td','select']):
        tag.decompose()
    # Remove all comments from the HTML
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    # remove all class attributes
    for tag in soup.find_all():
        tag.attrs = {key: value for key, value in tag.attrs.items() if key not in ['class', 'id','style','onclick']}
    # remove all empty tags
    for tag in soup.find_all():
        if not tag.contents and not tag.name == 'br':
            tag.extract()
    
    # Get the updated HTML
    clean_html = str(soup)
    minified_html = htmlmin.minify(clean_html, remove_empty_space=True)
    html_string = minified_html.replace('"', "'")

    print('=> 4.Asking openAI to do cool completion stuff now 🤖')
    response = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=[
            {"role": "user", "content": f"I have the following extracted HTML code for a webpage containing information about an online course ```{html_string}``` Kindly extract values pertaining to 'about the course', 'intended audience', 'industry support', 'youtube video url','course_layout' and 'prerequisites' for this course, from this HTML data. If you're unsure about a field's value, have it as empty string BUT get the other fields properly. The 'course_layout' information should be an array of weekly topics covered. Return ONLY a JSON object of the following form containing the relevant information for the keys:object:{{title, aboutTheCourse, instructorName, instructorBio, youtubeUrl, prerequisites, intendedAudience, industrySupport, courseLayout}}. Return nothing else in response except a JSON object"},
        ]
    )
    print('=> 5.Received response from AI agent 😏 - ')
    with open('processing.log', 'a') as f:
        f.write(f"Processed {url} with - Prompt({response.usage.prompt_tokens}) and Completion({response.usage.completion_tokens}). Total token usage Count - {response.usage.total_tokens} \n")
        # f.write(json.dumps(response))
    dataJsonString = response.choices[0].message.content.replace("\\", "").replace("\n", "")
    dataJson = json.loads(dataJsonString)

    print('=> 6.Formatted AI response object now 💭')
    infoObj = {
        'title': dataJson['title'],
        'about_the_course': dataJson['aboutTheCourse'],
        'instructor_name': dataJson['instructorName'],
        'instructor_bio': dataJson['instructorBio'],
        'youtube_url': dataJson['youtubeUrl'],
        'prerequisites': dataJson['prerequisites'],
        'intended_audience':  dataJson['intendedAudience'],
        'industry_support': dataJson['industrySupport'],
        'course_layout': dataJson['courseLayout'],
        
    }
    return infoObj

    ## Older approach using beautiful soup search and filtering - did not work 
    # with open('html.txt', 'w') as file:
    #     file.write(html_string)
    # intended_audience = find_text(soup,'intended audience')
    # industry_support = find_text(soup,'industry support')
    # prerequisites = find_text(soup,'prerequisites')

    # youtube_url = ''
    # youtube_element = soup.find(
    #     'iframe', src=lambda x: x and 'youtube.com' in x)
    # if youtube_element:
    #     youtube_url = youtube_element['src']


    # return intended_audience, industry_support, prerequisites, youtube_url
# With this change, the script will now search for the specified keywords within 'span' elements on the HTML page.

def append_row_to_csv(row, csv_file, index=None):
    # Check if the CSV file exists, if not, create it with the header (column names)
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print('🧩 CREATING INCREMENTAL CSV FILE NOW')
        with open(csv_file, 'w') as f:
            header = ','.join(row.keys()) + '\n'
            f.write(header)
        df = pd.read_csv(csv_file)

    print('🧩 CREATING DATAFRAME FROM INPUT SERIES ROW', json.dumps(row))
    # Convert the input Series row to a DataFrame
    new_row = row.to_frame().T

    print('🧩 CONCATENATING NEW ROW TO EXISTING DF')
    # Concatenate the existing DataFrame with the new row
    df = pd.concat([df, new_row], axis=0, ignore_index=True)

    print('🧩 CONVERTING BACK DF TO CSV')
    # Write the updated DataFrame to the CSV file
    df.to_csv(csv_file, index=False)

def append_dict_to_csv(data, csv_file):
    # Check if the CSV file exists, if not, create it with the header (dictionary keys)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the header (dictionary keys) if the file is new
        if not file_exists:
            writer.writerow(data.keys())

        # Write the row (dictionary values)
        writer.writerow(data.values())

def process_csv_chunk(chunk):
    with open('error.log', 'w') as f:
        for i, url in enumerate(chunk.iloc[:,16]):
            # selected_row = urls.loc[urls['Click here to join the course'] == url].idxmax()
            # selected_row_index = np.argmax(urls['Click here to join the course'] == url)
            # print(selected_row_index)
            # uid = selected_row.iloc[0,0]
            print(url)
            updated_url = url + '/preview'
            try:
                # Make an API request using the value
                print(f"--- PERFORMING OPERATION ON ROW {i}")
                print(f"=> 1. Scraping data from url - {updated_url}")
                scraped_info = scrape_data(updated_url);
                print("Here's the motherfucking data - ", scraped_info,"🔥")
                last_slash_index = url.rfind('/')
                # Slice the string to get the part after the last slash
                course_id = url[last_slash_index + 1:]
                new_attributes = {
                    'id':course_id,
                    'intended_audience':scraped_info['intended_audience'],
                    'instructor_name':scraped_info['instructor_name'],
                    'instructor_bio':scraped_info['instructor_bio'],
                    'about_the_course':scraped_info['about_the_course'],
                    'course_layout':';'.join(scraped_info['course_layout']),
                    'prerequisites':scraped_info['prerequisites'],
                    'industry_support':scraped_info['industry_support'],
                    'youtube_url':scraped_info['youtube_url'],
                    'course_url': updated_url

                }
                # Check if the unique ID exists in the DataFrame
                if (chunk['Click here to join the course'] == url).any():
                    # Get the index of the row with the specified unique identifier
                    row_index = (chunk['Click here to join the course'] == url).idxmax()
                    print(url, row_index)
                    # Update the row with new attributes
                    for key, value in new_attributes.items():
                        chunk.loc[row_index, key] = value
                    # Get the updated row using the index
                    print('🧩 FEEDING DICT DATA INTO CSV FILE =', row_index)
                    # updated_row = chunk.iloc[row_index]
                    # Write the updated row to CSV file
                    append_dict_to_csv(new_attributes, 'incremental.csv')
                    
                    print("👍 Row updated with new attributes:")
                else:
                    print("Unique ID not found in the DataFrame.")
            except Exception as e:
                # Write the error to the log file
                f.write(f'🚨 Error processing value {url}: {str(e)}\n')
                continue
            print(f'✅ Successfully scraped url' + updated_url + '\n' + '----------' + '\n')
    return chunk
# Open a log file for writing errors

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

curr_url = 'https://testbucket1841.s3.ap-south-1.amazonaws.com/csv-dump/pending.csv'
file_path = curr_url
df = pd.read_csv(file_path)
print("🐍 Length of the dataframe currently being looked at is -",len(df))
df.head()

#### Setting up base for parallelizing operations over this CSV data
def parallelize_dataframe_processing(df, func, num_partitions, num_workers):
    chunk_size = len(df) // num_partitions
    chunks = [df.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_partitions)]

    # If there are any leftover rows, add them to the last chunk
    if len(df) % num_partitions != 0:
        chunks[-1] = pd.concat([chunks[-1], df.iloc[num_partitions * chunk_size:]])

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_chunks = list(executor.map(func, chunks))

    return pd.concat(processed_chunks)

num_partitions = 7
num_workers = 7

def main():
    processed_df = parallelize_dataframe_processing(df, process_csv_chunk, num_partitions, num_workers)

    # Save the processed DataFrame to a new CSV file
    processed_df.to_csv('processed_file.csv', index=False)

if __name__ == '__main__':
    main()