import os
import shutil
import openai
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Scraping from website

def scraper(postId, basicUrl):
    print('Extracting data from site')

    posts = []

    imgDir = 'images'

    if os.path.exists(imgDir):
        shutil.rmtree(imgDir)

    os.mkdir(imgDir)

    postId = 182597
    pageIndex = 0

    imgIndex = 1

    while True:
        print('Scraping page', pageIndex)
        pageUrl = urljoin(basicUrl, f'?page={pageIndex}&action=loadmore&post_id={postId}&paged={pageIndex}&utmSource=&isMobile=False')
        
        resp = requests.get(pageUrl)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            title = soup.find('h2')

            if title is None:
                break

            paragraphs = title.find_next_siblings('p')
        
            description = paragraphs[0].text.strip()
                
            images = [img['src'] for img in soup.findAll('img')]

            if len(paragraphs) < 2:
                subdescription = ''
            else:
                subdescription = paragraphs[1].text.strip()

            row = {
                'title': title.text.strip(),
                'description': description,
                'image': images[-1],
                'subdescription': subdescription
            }

            imgFilename = f"{imgIndex}.{row['image'].split('/')[-1].split('.')[-1]}"
            
            with open(f'{imgDir}/{imgFilename}', 'wb') as imgFile:
                imgFile.write(requests.get(row['image']).content)
                imgIndex += 1

            posts.append(row)

        pageIndex += 1

    return posts

# Replacer
def worker(post, query_text):
    title = post['title']
    print(title, '==================')
    newDescription = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{post['description']},{query_text}",
            temperature=0.78,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )['choices'][0]['text'].strip()

    newSubDescription = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"{post['subdescription']},{query_text}",
                temperature=0.54,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )['choices'][0]['text'].strip()
    
    replacedPost = post
    replacedPost['description'] = newDescription
    replacedPost['subdescription'] = newSubDescription

    print(replacedPost)

    return replacedPost
        
# Replacing with ai 

