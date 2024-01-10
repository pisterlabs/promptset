import openai, os, sys
import requests
from bs4 import BeautifulSoup

openai.api_key = os.getenv("OPENAI_API_KEY")

paragraphAmount = 5
# Includes intro paragraph

def extract(aiContent):
    return str(aiContent["choices"][0]["text"])

def isValid(topic):
    valid = extract(openai.Completion.create(
    model="text-davinci-002",
    prompt=f"Respond with yes or no. Is {topic} a topic thats valid for an article and safe for children to learn about?",
    temperature=0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )).lower().replace("\n", "")
    if valid[:2] == 'ye':
        return True
    return False

def topicFormatter(topic):
    topic = topic.replace("\n", "")
    newTopic = openai.Edit.create(
    model="text-davinci-edit-001",
    input=topic,
    instruction="1. Fix the spelling (English)\n2. Capitalize in the style of a title\n3. Make the text singular, not plural",
    temperature=0,
    top_p=1
    )
    return extract(newTopic).replace("\n", "")

def createImages(topic, subheadings):

    searchFormat = lambda search : search.replace(" ", "+")
    URL = lambda search : f"https://bing.com/images/search?q={search}&qft=+filterui:aspect-wide+filterui:licenseType-Any&safeSearch=strict"

    headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
    }
    result = []
    
    for subheading in subheadings:
        query = f"{topic} {subheading}" if subheading not in ['Introduction', 'Conclusion'] else topic
        searchURL = URL(searchFormat(query))
        page = requests.get(searchURL, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        images = soup.find_all('img', class_='mimg')
        i = 0
        added = False
        for image in images:
            if not image.has_key('src'):
                continue
            src = image['src'].split('?')[0]
            if "data:image" in src or 'gif' in src or ';base64' in src: #avoid pure data images
                continue
            if src in result:
                continue
            else:
                result.append(src)
                added = True
                break
        if added == False:
            result.append("#") #placeholder for empty image
    return result


def createWiki(topic, paragraphAmount):

    # test = 'test'

    topic = topicFormatter(topic)
    
    if not isValid(topic):
        print("Page invalid or inappropriate")
    #Check page exists
    elif not os.path.exists("/var/www/html/pages/" + topic.lower() + ".data"):
        # Create subheadings
        subheadings = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Give {paragraphAmount} one-word subheadings, in numbered dot points, that would be found in an informative article about {topic}. Make the first subheading an introduction to the topic.",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        subheadings = extract(subheadings)

        subheadings = subheadings.split("\n")
        subheadings = [e[3:] for e in subheadings]

        while "" in subheadings:
            subheadings.remove("")

        # Create paragraphs for the 5 subheadings
        paragraphs = []
        for subheading in subheadings:
            paragraphPrompt = f"Create a single paragraph for an informative article. The overall topic of the article is {topic}. The subtopic for the paragraph to create is {subheading}.\n\nThe purpose of the article is to educate elementary school students. The paragraph needs to be understandable, informative and accurate. Attempt to completely cover all aspects of the subtopic, while being easy to follow. Focus on more generally accepted information. increase the length of the paragraph by elaborating on related topics.",
            paragraph = openai.Completion.create(
            model="text-davinci-002",
            prompt=paragraphPrompt,
            temperature=0,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            paragraphs.append("\n\=====/\n***" + subheading + extract(paragraph))

        # Create images
        images = createImages(topic, subheadings)

        #Format file
        new_text = "#Header Data\n#Title:\n" + topic + \
                "\n#Heading\n" + topic + \
                "\n#Article Created By GPT3\n#\======/"
        for paragraph in paragraphs:
            paragraph.replace("\n\n\n", "\n")
            paragraph.replace("\n\n", "\n")
            new_text += paragraph
        new_text += "\n\======/"
        for image in images:
            new_text += ("\n" + image)
        new_text += "\n"
        
        #Save file
        f = open("/var/www/html/pages/" + topic.lower() + ".data", "w")
        f.write(new_text)
        f.close()
        print ("Success: " + topic.lower())
    else:
        print("Page exists: " + topic.lower())


if len(sys.argv) == 2:
    createWiki(sys.argv[1], paragraphAmount)
else:
    print("Bad args")
