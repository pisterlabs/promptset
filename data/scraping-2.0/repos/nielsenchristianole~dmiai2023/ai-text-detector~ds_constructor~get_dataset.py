# print(">>> Importing libraries...")
# from datasets import load_dataset

# print(">>> Loading dataset...")
# dataset = load_dataset("wikipedia", "20220301.da", split="train", beam_runner="DirectRunner")

# print(dataset[0])   

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def clean_text(text):
    # Remove all non-alphanumeric characters and all special unicode characters
    # Keep: .,;:!?-$%&/'" and danish characters along with spaces
    # text = re.sub(r'[^a-zA-Z0-9æøåÆØÅ.,;:!? \-$/\'"%&]', ' ', text)
    
    # Remove all double spaces and links
    # text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'http\S+', '', text)
    
    return text

def extract_data(urls, min_length = 20, max_length = 1000):
    extracted = []
    
    skipped_bodies = 0
    skipped_articles = 0
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                         "Version/15.4 Safari/605.1.15"}
    
    for url in tqdm(urls):
        # Send a HTTP request to the URL of the webpage
        response = requests.get(url, headers=headers)
        print(f'Status code: {response.status_code}')

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        content = {'header': [], 'summary': [], 'body': []}
        
        ### EXTRACT HEADERS ###
        heads = soup.find_all('h1', class_='article__title headline headline--mega') # politiken
        
        div_temp = soup.find('div', class_='article__header') # bt
        heads += div_temp.find_all('h1') if div_temp is not None else [] # bt
        
        div_temp = soup.find('div', class_='header-elements') # information
        heads += div_temp.find_all('h1') if div_temp is not None else [] # information
        
        heads += soup.find_all('h1', class_='article-header__title') # berlinske
        
        div_temp = soup.find('div', class_='tc_page__header') # tv2
        heads += div_temp.find_all('h1') if div_temp is not None else [] # tv2
        
        ### EXTRACT SUMMARIES ###
        summaries = soup.find_all(class_='summary__p') # politiken
        
        summaries += soup.find_all('div', class_='field field-name-field-underrubrik') # information
        
        summaries += soup.find_all('p', class_='article-header__intro') # berlinske
        
        ### EXTRACT BODIES ###
        bodies = soup.find_all(class_='body__p') # politiken
        
        div_temp = soup.find('div', class_='article-content') # bt
        bodies += div_temp.find_all('p') if div_temp is not None else [] # bt
        
        div_temp = soup.find('div', class_='field field-name-body') # information
        bodies += div_temp.find_all('p') if div_temp is not None else [] # information
        
        div_temp = soup.find('div', class_='article-body') # berlinske
        bodies += div_temp.find_all('p') if div_temp is not None else [] # berlinske
        
        div_temp = soup.find('div', class_='tc_richcontent') # tv2
        bodies += div_temp.find_all('p') if div_temp is not None else [] # tv2
        
        div_temp = soup.find('div', class_='row justify-center') # verdens bedste
        heads += div_temp.find_all('p') if div_temp is not None else [] # verdens bedste
        
        # For each header, summary and body, extract the text and store it in the dictionary
        for header in heads:
            content['header'].append(clean_text(header.text))
        for summary in summaries:
            content['summary'].append(clean_text(summary.text))
        for body in bodies:
            if len(body.text.split()) < min_length or len(body.text.split()) > max_length:
                if len(content['body']) != 0:
                    content['body'][-1] += '\n\n' + clean_text(body.text)
                else:
                    skipped_bodies += 1
                continue
            content['body'].append(clean_text(body.text))
            
        if content['header'] == [] or content['body'] == []:
            print("Skipping [missing header or body]:", url)
            skipped_articles += 1
            continue    
        
        # If there are more than 2 bodies, 10% change to merge two sequential bodies
        if len(content['body']) > 2 and np.random.rand() < 0.2:
            idx = np.random.randint(0, len(content['body'])-2) # -2 since the last should not be merged with the first
            #merge idx and idx+1
            content['body'][idx] += '\n\n' + content['body'][idx+1]
            del content['body'][idx+1]
        
        extracted.append(content)

    # Save the extracted content as a .npz file
    np.savez('ds_constructor/dataset.npz', extracted=extracted)
    
    print("Skipped bodies:", skipped_bodies)
    print("Skipped articles:", skipped_articles)
    print("Keept articles:", len(extracted))
    
    print("Extraction complete!")
    
def get_urls(path = 'ds_constructor/websites.txt'):
    # Read all URLs from websites.txt (each line is a URL)
    with open(path, 'r', encoding='utf-8') as f:
        urls = f.readlines()
        
    # Remove duplicates
    urls = list(set(urls))
    
    return urls

def convert_scraped_to_text():
    # Read the dataset.npz file and count the number of headlines, summaries and bodies
    dataset = np.load('ds_constructor/dataset.npz', allow_pickle=True)['extracted']
    bodies_all = []
    for d in dataset:
        bodies_all += d['body']
        
    # Save the extracted content as a .txt file
    with open('ds_constructor/scraped_text.txt', 'w', encoding='utf-8') as f:
        for body in bodies_all:
            f.write(body)
            f.write('\n---\n')
            
def convert_val_to_text():
    data_val = pd.read_csv('data/val_data.tsv', sep='\t', header=None, names=['text'])
    
    with open('data/val_text.txt', 'w', encoding='utf-8') as f:
        for text in data_val['text']:
            f.write(text)
            f.write('\n---\n')
    
def convert_gen_and_scrap_to_csv():
    # Read the generated_text.txt and scraped_text.txt files
    # Each entry is seperated by '\n---\n'
    with open('ds_constructor/generated_text.txt', 'r', encoding='utf-8') as f:
        generated = f.read().split('\n---\n')
    with open('ds_constructor/scraped_text.txt', 'r', encoding='utf-8') as f:
        scraped = f.read().split('\n---\n')
    # Remove empty entries
    generated = [g for g in generated if g != '']
    scraped = [s for s in scraped if s != '']
    # clean the text
    generated = [clean_text(g) for g in generated]
    scraped = [clean_text(s) for s in scraped]
    
    # Create a dataframe with the generated and scraped text
    # Add a label column with 1 for generated and 0 for scraped
    df = pd.DataFrame({'text': generated + scraped, 'is_generated': [1]*len(generated) + [0]*len(scraped)})
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the dataframe as a .csv file
    df.to_csv('ds_constructor/data_custom.csv', index=False)
    
urls = get_urls()
extract_data(urls)
convert_scraped_to_text()
convert_val_to_text()
convert_gen_and_scrap_to_csv()

# Read the dataset.npz file and count the number of headlines, summaries and bodies
dataset = np.load('ds_constructor/dataset.npz', allow_pickle=True)['extracted']
bodies_all = []
headlines_all = []
for d in dataset:
    bodies_all += d['body']
    headlines_all += d['header']

print("Number of headlines:", sum([len(d['header']) for d in dataset]))
print("Number of summaries:", sum([len(d['summary']) for d in dataset]))
print("Number of bodies:", sum([len(d['body']) for d in dataset]))

# Get length distribution of all individual segments in all bodies
lengths_scrape = []
for d in dataset:
    for body in d['body']:
        lengths_scrape.append(len(body.split()))

# load data.csv from data
data_test = pd.read_csv('data/data.csv')
data_val = pd.read_csv('data/val_data.tsv', sep='\t', header=None, names=['text'])
data_custom = pd.read_csv('ds_constructor/data_custom.csv')

# Get the length of each datapoint in dataset['text']
length_val = []
length_custom = []
length_gen = []
for text in data_val['text']:
    length_val.append(len(text.split()))
for text in data_custom['text']:
    length_custom.append(len(text.split()))
for text in data_custom[data_custom['is_generated'] == 1]['text']:
    length_gen.append(len(text.split()))
    
# Plot the length distribution of all individual segments in all bodies
# Normalized by the total number of segments
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

for i, (length_data, name) in enumerate(zip([lengths_scrape, length_val, length_custom, length_gen],
                                    ['Scraped data', 'Validation data', 'Custom data', 'Generated data'])):
    axs[i].hist(lengths_scrape, bins=100, density=True)
    axs[i].set_title(name)
    axs[i].set_xlim(0, 400)
    axs[i].vlines(x=np.mean(length_data), color='r', ymin=0, ymax=0.02)
    axs[i].vlines(x=np.median(length_data), color='g', ymin=0, ymax=0.02)
    axs[i].vlines(x=np.min(length_data), color='b', ymin=0, ymax=0.02)
    axs[i].vlines(x=np.max(length_data), color='b', ymin=0, ymax=0.02)
plt.show()



from openai import OpenAI

client = OpenAI(api_key='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')

def generate_article_data():
    models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
    model_idx = 1

    people = [
        "Du er en meget professionel og vel formuleret journalist med mange års erfaring. Du skriver på dansk.",
        "Du er en dygtig journalist med medium erfaring. Du skriver på dansk.",
        "Du er en nyansat journalist med lidt erfaring. Du skriver på dansk.",
    ]

    content = [
        " Du skriver artikler der fokusere på samfundets påvirkning/indflydelse",
        " Du skriver politisk prægede artikler",
        " Du har en meget stærk faglig viden inden for emnet som du benytter i dine artikler",
        " Du elsker at quote folk der har stor indflydelse på området",
        " Du benytter dig af egne holdninger og erfaringer i dine artikler",
        " Du er generelt meget kritisk over for emnet",
        " Du laver linjeskift i dine artikler",
        ]

    length = ["5 til 15", "5 til 20", "15 til 25", "20 til 40"]


    # Pick a random person and content
    person = np.random.choice(people)
    content = np.random.choice(content)
    length = np.random.choice(length)
    headline = np.random.choice(headlines_all, 3)
    number = np.random.randint(5, 10)
    model_idx = np.random.randint(0, 2)

    print(">> person and content:")
    print(person+content)
    print(">> length:", length)
    print(">> model:", models[model_idx])
    print(">> headlines:")
    print(headline,"\n------------------\n")


    completion = client.chat.completions.create(
    model=models[model_idx],
    messages=[
        {"role": "system", "content": person + content},
        {"role": "user", "content": f"Skriv {number} små udpluk fra dine danske artikler.\
                                    Hvert udpluk skal ikke (nødvendigvis) komme fra samme artikel og hvert udpluk bør ikke være sammenhængende.   \
                                    Includer eventuelt et quote fra en bruger eller en ekspert. Du må gerne inkludere specifikke detaljer. \
                                    Hvert udpluk skal være mellem {length} sætninger. \
                                    Overvej også at inkludere udpluk der virker totalt ude af kontekst.\
                                    Overskriften de fiktive artikler du tager udgangs i er: {headline}\
                                    Hvis en overskrift er voldelig eller stødende så ignorer den.\
                                    Udplukkene vil bruges til undevisningsrettet formål og vil ikke blive offentliggjort. \
                                    Overvej flere sektioner i hvert udpluk. \
                                    Seperer hvert udpluk med 3 dashes, e.i. '---'. Giv dem IKKE overskrift\
                                    Tak for hjælpen"}
    ]
    )

    print(completion.choices[0].message.content)

    # Save the generated text to a .txt file
    content = completion.choices[0].message.content

    # If the generated text is less 1000 characters discard it
    if len(content) < 700:
        print("Generated text is less than 700 characters. Discarding...")
    else:
        # Add the generated text to the generated_text.txt file
        with open('ds_constructor/generated_text.txt', 'a', encoding='utf-8') as f:
            f.write(content)
            f.write('\n---\n')
            
# for i in tqdm(range(50)):
#     generate_article_data()