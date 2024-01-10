from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
Client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL")

def upload_file(file_path, purpose):
    with open(file_path, "rb") as file:
        response = Client.files.create(file=file, purpose=purpose)
    return response.id


def methodology(keyword):
    systemmsg = f"You are a methodology section generator for {keyword} ranking its items and or categories. Output only the text that will be used in article."
    messages = list()
    messages.append({"role": "system", "content": systemmsg})
    prompt = f"Generate a methodology section in html starting at h2 for a wordpress article titled {keyword}. Include a heading for the section."
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    response_message = response.choices[0].message.content
    # print(response_message)
    return response_message


def introduction(article):
    systemmsg = f"You are an introduction section generator for wordpress articles. You generate a very short introduction"
    messages = list()
    messages.append({"role": "system", "content": systemmsg})
    prompt = f"Generate a one paragraph introduction without including the methodology for a wordpress article format it html starting at h2: \n {article}"
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    response_message = response.choices[0].message.content
    # print("\n\nIntroduction:",response_message)
    return response_message



def read_items(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['items']


#TODO change this to be an assistant API call to add internal links and to be sure it doesnt max out of tokens.
def generate_sections(methodology,keyword,items):
    rated_items = f"<h2> {keyword} </h2>\n\n"
    messages = list()
    systemmsg = f"You are a section generator for wordpress articles. Write in a journalist tone and based off: \n {methodology}."
    messages.append({"role": "system", "content": systemmsg})
    for item in items:
        name = item['Title']
        link = item['URL']
        photo = item['Image URL']
        prompt = f"Generate a short one paragraph section in html about {name} for the article title {keyword}. Be sure to add their link whenever you mention their name: {link} and show the image if one: {photo}. Dont add any headers."
        messages.append({"role": "user", "content": prompt})
        response = Client.chat.completions.create(
            model=model,
            messages=messages,
        )
        response_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_message})
        # print(response_message)
        rated_items += f"<h3>{name}</h3>\n{response_message}\n\n" 
    return rated_items


def overview(keyword, rated_items):
    systemmsg = f"You are an article overview generator for wordpress articles. You generate the overview with this format: \n <h2>{keyword}</h2>: \n <ul> \n <li> <a href='https://www.wired.com/'>Wired</a> </li> \n </ul> "
    messages = list()
    messages.append({"role": "system", "content": systemmsg})
    prompt = f"Generate an overview of this article with no images in html for the article titled {keyword}. Keep it one short sentence MAX for each section: {rated_items}."
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    response_message = response.choices[0].message.content
    return response_message

def table_of_contents(article):
    systemmsg = f"You are an table of contents generator for wordpress articles. You generate the table of contents with this format: <h2> Table of Contents </h2> \n <ul> \n <li> <a href='#introduction'>Introduction</a> </li> \n </ul>..."
    messages = list()
    messages.append({"role": "system", "content": systemmsg})
    prompt = f"ONLY generate the table of contents for this article in html with links to headings, include a heading for the section: {article}."
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    response_message = response.choices[0].message.content
    return response_message


def generate_json(keyword,methodology):
    systemmsg = f"You are a json generator for {keyword} ranking its items and or categories. You use: {methodology} Output JSON."
    messages = list()
    messages.append({"role": "system", "content": systemmsg})
    prompt = f'Create a list of {keyword} and links to websites. Leave the image URL and Image URL blank like this JSON: {{"items": [{{"Title": "TechCrunch", "URL": "", "Image URL": ""}},...]}} '
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model='gpt-4-1106-preview',messages=messages,response_format={ "type": "json_object" })
    response_message = response.choices[0].message.content
    # print(response_message)
    return response_message


def autoblog(keyword,methodology_):
    # methodology_ = methodology(keyword)
    items = read_items('data.json')
    sections = generate_sections(methodology_,keyword,items)
    overview_ = overview(keyword,sections)
    article =  methodology_ + "\n\n"+ sections
    introduction_ = introduction(article)
    article += "\n\n"+ introduction_ +"\n\n" + overview_ + "\n\n" + methodology_ +"\n\n"+ sections
    table_of_contents_ = table_of_contents(article)
    final_article = introduction_ +"\n\n"+ table_of_contents_ +"\n\n"+ overview_ + "\n\n" + methodology_ +"\n\n\n"+ sections
    # print(final_article)
    #replace markdown tags with nothing
    final_article = final_article.replace("“`html","")
    final_article = final_article.replace("“`","")
    final_article = final_article.replace("```html","")
    final_article = final_article.replace("```","")
    final_article = final_article.replace('"','')
    #add results to results.md file
    with open('results.md', 'w') as file:
        file.write(final_article)
    return final_article



def seo(article):
    systemmsg = "You are an SEO generator for wordpress articles. You return only the text that will be used. e.g. response: Top Tech Publications"
    messages = list()

    messages.append({"role": "system", "content": systemmsg})
    prompt = f"Heres the article:\n {article}."
    messages.append({"role": "user", "content": prompt})

    prompt = f"Generate the Focus keyphrase for this article."
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    focus_keyphrase = response.choices[0].message.content
    focus_keyphrase = focus_keyphrase.replace('"','')
    messages.append({"role": "assistant", "content": focus_keyphrase})


    prompt = f"Generate the title for this article"
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    title = response.choices[0].message.content
    title = title.replace('"','')
    messages.append({"role": "assistant", "content": title})

    prompt = f"Generate the SEO title for this article"
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    seo_title = response.choices[0].message.content
    seo_title = seo_title.replace('"','')
    messages.append({"role": "assistant", "content": seo_title})

    prompt = f"Generate a meta description for this article in one very short sentence"
    messages.append({"role": "user", "content": prompt})
    response = Client.chat.completions.create(model=model,messages=messages,)
    meta_description = response.choices[0].message.content
    meta_description = meta_description.replace('"','')
    messages.append({"role": "assistant", "content": meta_description})

    return title, focus_keyphrase, meta_description, seo_title






#TODO scrape google maps for JSON data- may be different for style of application

#TODO def generate_ranking(methodology): 
    #prompt: generate ranking for {category} based off {methodology} in JSON format: {category: {publication: {rank: 1, link: https://www.wired.com/, photo: https://www.wired.com/logo.png}}}
    #will give it the doc using assistants API. 

#TODO add assitants API to take advantage of files with our interal links to add to the sections and use of thread so we dont max out tokens.

#TODO have a grading GPT that states if the article is good to post or not. If not, it will return a list of things to fix. And then call a GPT to fix the section.

#note Maybe the section builder shoudlnt have access to the image url and we do that part manuely. 
    #if img_url :
        #add the image
    #else:
        #dont add the image
    
#TODO change from mardown to HTML for the final article.