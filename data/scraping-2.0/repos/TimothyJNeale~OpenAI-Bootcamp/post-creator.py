import openai
import os
import shutil
import requests

from dotenv import load_dotenv
from git import Repo
from bs4 import BeautifulSoup as Soup
from pathlib import Path

WORKING_PATH = "/Users/sealislandmedia/Desktop/TimothyJNeale.github.io"
PATH_TO_BLOG_REPO = Path(os.path.join(WORKING_PATH, ".git"))
PATH_TO_BLOG = PATH_TO_BLOG_REPO.parent

PATH_TO_CONTENT = PATH_TO_BLOG/"content"
PATH_TO_CONTENT.mkdir(exist_ok=True, parents=True)


 ############# Helper functions #############
def update_blog(commit_message='Update blog'):
    repo = Repo(PATH_TO_BLOG_REPO)

    repo.git.add(all=True)
    repo.index.commit(commit_message)

    origin = repo.remote(name='origin')
    origin.push()



def create_post(title, content, cover_image=None):
    cover_image = Path(cover_image)

    files = len(list(PATH_TO_CONTENT.glob("*.html")))
    new_title = f"{files+1}.html"
    path_to_new_post = PATH_TO_CONTENT/new_title

    shutil.copy(cover_image, PATH_TO_CONTENT)

    if not os.path.exists(path_to_new_post):
        with open(path_to_new_post, "w") as f:
            f.write('<DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('<meta charset="utf-8">\n')
            f.write('<meta name="viewport" content="width=device-width, initial-scale=1">\n')
            f.write(f'<title>{title}</title>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            f.write(f"<img src='{cover_image.name}' alt='Cover Image'> <br /> \n")
            f.write(f'<h1>{title}</h1>\n')
            f.write(content.replace('\n', '<br /> \n'))
            f.write('</body>\n')
            f.write('</html>\n')

            #print(f"Created new post: {path_to_new_post}")
            return path_to_new_post
        
    else:
        raise FileExistsError("File already exists")



# Check for duplicate links
def check_for_duplicate_links(path_to_new_content, links):
    urls = [str(link.get("href")) for link in links]
    content_path = str(Path(*path_to_new_content.parts[-2:]))
    return content_path in urls



def write_to_index(path_to_new_content):
    with open(PATH_TO_BLOG/"index.html") as index:
        soup = Soup(index.read(), features="html.parser")
    
    links = soup.find_all("a")
    last_link = links[-1]

    if check_for_duplicate_links(path_to_new_content, links):
        raise ValueError("Link already exists in index")
    
    link_to_new_blog = soup.new_tag("a", href=Path(*path_to_new_content.parts[-2:]))
    link_to_new_blog.string = path_to_new_content.name.split(".")[0]
    last_link.insert_after(link_to_new_blog)

    with open(PATH_TO_BLOG/"index.html", "w") as f:
        f.write(str(soup.prettify(formatter="html5")))


def create_prompt(title):
    prompt = f"""
    About this blog : This blog is for people with a non-technical background. It hopes to teach and inform readers
    about AI in general and generative AI in particular.

    I am writing a blog post about {title}.
    tags: AI, Generative AI, GPT-3, OpenAI, Machine Learning, Deep Learning, Neural Networks, AGI

    Full Text :
    """
    return prompt

def dalle2_prompt(title):
    prompt = f"""A Professional photograph showing '{title}' as a concept. 
    # It should include real objects such as servers or smart phones.
    # 15mm, studio lighting, 1/125s, f/5.6, ISO 100, 5500K, 1/4
    # Do not include the title."""
    return prompt

# Download and save the image returned from DALLE
def save_image(image_response, filename):
    image_url = image_response['data'][0]['url']

    image_res = requests.get(image_url, stream=True)
    if image_res.status_code == 200:
        with open(filename, 'wb') as image_file:
            shutil.copyfileobj(image_res.raw, image_file)
    else:
        print('Image couldn\'t be retreived')

    return image_res.status_code


############# Execution code starts here #############
title = "The Future of AI"
prompt = create_prompt(title)
image_prompt = dalle2_prompt(title)
cover_image = f"dev/{title}.jpg" 

# load environment variables from .env file
load_dotenv()

# get api key from environment variable
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    max_tokens=1000,
                                    temperature=0.7)

blog_content = response['choices'][0]['text']
print(blog_content)

blog_image = openai.Image.create(prompt=image_prompt,
                                    n=1,
                                    size='1024x1024')
save_image(blog_image, cover_image)

path_to_new_post = create_post(title, blog_content, cover_image)
with open(PATH_TO_BLOG/"index.html") as index:
    soup = Soup(index.read(), features="html.parser")
#print(str(soup))

write_to_index(path_to_new_post)
update_blog(commit_message='Update blog')


