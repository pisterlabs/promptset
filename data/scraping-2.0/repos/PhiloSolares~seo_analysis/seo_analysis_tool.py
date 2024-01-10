import os
import requests
from bs4 import BeautifulSoup, Tag
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import gradio as gr
import openai
from googlesearch import search
from pytrends.request import TrendReq
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.display import HTML
import numpy as np
import matplotlib.cm as cm
from urllib.parse import urlparse, urljoin
import os
from celery import Celery



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

# Set your OpenAI API key here
openai.api_key = os.environ['OPENAI_API_KEY']


#@title Define functions

def get_image_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return '<img src="data:image/png;base64,{}"/>'.format(base64.b64encode(buf.getvalue()).decode('ascii'))


def search_top_competitors(keywords, num_results=10):
    competitors = set()
    for keyword in keywords:
        for url in search(keyword, num_results=num_results):
            competitors.add(url)
    return list(competitors)



def get_page_content(url):
    response = requests.get(url)
    return BeautifulSoup(response.text, 'html.parser')

def get_meta_tags(soup):
    meta_tags = soup.find_all('meta')
    return {tag.get('name'): tag.get('content') for tag in meta_tags if tag.get('name')}

def get_heading_tags(soup):
    headings = {}
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        headings[tag] = [heading.text for heading in soup.find_all(tag)]
    return headings

def analyze_keywords(keywords_counter, top_n=10):
    return keywords_counter.most_common(top_n)

def visualize_keywords(keywords_counter, top_n=10):
    common_keywords = analyze_keywords(keywords_counter, top_n)
    df = pd.DataFrame(common_keywords, columns=['Keyword', 'Count'])
    df.set_index('Keyword', inplace=True)
    df.plot(kind='bar', figsize=(12, 6))
    plt.title('Top Keywords')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    
    fig = plt.gcf()  # Get the current figure

    plt.tight_layout()
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_image_file.name, format='png')
    plt.close()
    return temp_image_file.name


def plot_trends(keywords):
  pytrends = TrendReq(hl='en-US', tz=360, retries=3)
  pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='', gprop='')
  trends_data = pytrends.interest_over_time()
  return trends_data



def preprocess_text(text, min_word_length=3):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if len(word) >= min_word_length and word not in stop_words]
    return words

def visualize_clusters(words, model):
    matrix = np.zeros((len(words), model.vector_size))

    for i, word in enumerate(words):
        matrix[i, :] = model.wv[word]

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    distance_matrix = 1 - cosine_similarity(matrix)
    coords = mds.fit_transform(distance_matrix)
    
    x, y = coords[:, 0], coords[:, 1]

    for i, word in enumerate(words):
        plt.scatter(x[i], y[i], alpha=0.5)
        plt.text(x[i], y[i], word, fontsize=10)

    plt.title('Word Clusters based on Thematic Relatedness')
    plt.show()



def create_cluster_table(words, model, clusters):
    matrix = np.zeros((len(words), model.vector_size))

    for i, word in enumerate(words):
        matrix[i, :] = model.wv[word]

    # Create a dictionary to store words per cluster
    cluster_dict = {}
    for i, word in enumerate(words):
        cluster_id = clusters[i]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(word)

    # Create a DataFrame from the dictionary
    max_words = max(len(cluster_words) for cluster_words in cluster_dict.values())
    num_clusters = len(cluster_dict)
    data = {f"Cluster {i}": cluster_dict.get(i, []) + [None] * (max_words - len(cluster_dict.get(i, [])))
            for i in range(num_clusters)}

    df = pd.DataFrame(data)
    return df


def clean_text(text):
    # Separate words that are meant to be separated
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove nonsensical words
    try:
      english_words = set(words)
    except:
      english_words = set(words.words())
    clean_tokens = [token for token in tokens if token.lower() in english_words or token.istitle()]

    # Join tokens back into a string
    clean_text = ' '.join(clean_tokens)

    return clean_text

def visualize_clusters_og(words, model):
    matrix = np.zeros((len(words), model.vector_size))

    for i, word in enumerate(words):
        matrix[i, :] = model.wv[word]

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix)

    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(matrix)

    x, y = coords[:, 0], coords[:, 1]

    colors = cm.rainbow(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(8, 8))
    for i, word in enumerate(words):
        plt.scatter(x[i], y[i], c=[colors[clusters[i]]], alpha=0.7)
        plt.text(x[i], y[i], word, fontsize=10)

    plt.xticks([])
    plt.yticks([])
    plt.title('Word Clusters based on Thematic Relatedness')
    plt.show()


def visualize_clusters_plot(words, model):
    matrix = np.zeros((len(words), model.vector_size))

    for i, word in enumerate(words):
        matrix[i, :] = model.wv[word]

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix)

    try:
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(matrix)
    except ValueError:
        max_perplexity = len(words) - 1
        tsne = TSNE(n_components=2, random_state=42, perplexity=max_perplexity)
        coords = tsne.fit_transform(matrix)


    x, y = coords[:, 0], coords[:, 1]

    colors = cm.rainbow(np.linspace(0, 1, n_clusters))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [sum(clusters == 0) + sum(clusters == 1), sum(clusters == 2) + sum(clusters == 3)], 'height_ratios': [sum(clusters == 0) + sum(clusters == 2), sum(clusters == 1) + sum(clusters == 3)]})
    fig.subplots_adjust(wspace=0, hspace=0)

    for ax in axs.ravel():
        ax.axis('off')

    for i, word in enumerate(words):
        cluster_idx = clusters[i]
        ax = axs[cluster_idx // 2, cluster_idx % 2]
        ax.scatter(x[i], y[i], c=[colors[cluster_idx]], alpha=0.7)
        ax.text(x[i], y[i], word, fontsize=10)

    plt.legend(loc="best", fontsize=13)
    plt.tight_layout()
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_image_file.name, format='png')
    plt.close()
    return temp_image_file.name, clusters


def sanitize_url(url):
    if not re.match('^(http|https)://', url):
        url = 'http://' + url
    
    if not re.match('^(http|https)://www\.', url):
        url = re.sub('^(http|https)://', r'\g<0>www.', url)
    
    return url



# Configure the Celery app
app = Celery("tasks", broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])

# Define the inputs and outputs
competitor_url_input = gr.inputs.Textbox(label="Competitor URL", placeholder="Enter a competitor URL")

full_site_scrape_checkbox = gr.inputs.Checkbox(label="Tick for full site scrape (otherwise landing page only)")

meta_tags_output = gr.outputs.Textbox(label="Meta Tags")
heading_tags_output = gr.outputs.Textbox(label="Heading Tags")
top10keywords_output = gr.outputs.Textbox(label="Top 10 Keywords")
cluster_table_output = gr.outputs.HTML(label="Cluster Table")
cluster_plot_output = gr.outputs.Image(type='filepath', label="Cluster Plot")
keyword_plot_output = gr.outputs.Image(type='filepath', label="Keyword Plot")
seo_analysis_output = gr.outputs.Textbox(label="SEO Analysis")


def append_unique_elements(source, target):
    for element in source:
        if isinstance(element, Tag) and element not in target:
            target.append(element)

def get_internal_links(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    internal_links = set()

    for link in soup.find_all("a"):
        href = link.get("href")

        if href:
            joined_url = urljoin(url, href)
            parsed_url = urlparse(joined_url)

            if parsed_url.netloc == urlparse(url).netloc:
                internal_links.add(joined_url)

    return internal_links

def analyze_single_page(competitor_url: str):
    sanitized_url = sanitize_url(competitor_url)
    soup = get_page_content(sanitized_url)

    # Scrape and analyze meta tags
    meta_tags = get_meta_tags(soup)
    topmetatags = ""
    for name, content in meta_tags.items():
        if "description" in name.lower():
            topmetatags += (f"{name}: {content}\n")

    # Scrape and analyze heading tags
    heading_tags = get_heading_tags(soup)
    topheadingtags = ""
    for tag, headings in heading_tags.items():
        filtered_headings = [heading for heading in headings if len(heading) > 2]
        if filtered_headings:
            topheadingtags += (f"{tag}: {', '.join(filtered_headings)}\n")

    # Scrape, analyze, and visualize keywords from page content
    page_text = soup.get_text()
    page_text_cleaned = clean_text(page_text)
    preprocessed_text = preprocess_text(page_text_cleaned)

    keywords_counter = Counter(preprocessed_text)
    top10keywords = ""

    for keyword, count in analyze_keywords(keywords_counter, top_n=10):
        top10keywords += (f"{keyword}: {count}\n")

    # Semantic clustering and visualization
    sentences = [preprocessed_text[i:i+10] for i in range(0, len(preprocessed_text), 10)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    words = [word for word, _ in analyze_keywords(keywords_counter, top_n=50)]
    clusters = [model.wv.doesnt_match(words)] * len(words)


    cluster_plot,clusters = visualize_clusters_plot(words, model)
    cluster_table = create_cluster_table(words, model, clusters)
    keyword_plot = visualize_keywords(keywords_counter, top_n=10)

    table_string = cluster_table.to_string(index=False)
    SEO_prompt = f"""The following information is given about a company's website:

      Meta Tags:
      {{meta_tags}}

      Heading Tags:
      {{heading_tags}}

      Top 10 Keywords:
      {{top10keywords}}

      The following table represents clusters of thematically related words identified using NLP and clustering techniques. Each column represents a different cluster, and the words in each column are thematically related.

      {table_string}

      Please analyze the provided information and perform the following tasks:
      1. Predict what the website is all about (the market sector).
      2. Based on the market sector of the company, give a name to each cluster based on the theme it represents. The name needs to be the best summary of all the words in the cluster.
      3. Perform a SWOT analysis (Strengths, Weaknesses, Opportunities, and Threats) from an SEO perspective for the company as a whole, taking into account the meta tags, heading tags, top 10 keywords, and the clusters.
      Please provide your analysis in a clear and concise manner.
      4. Lastly, suggest a list of 5 single words and 5 phrases (no longer than 3 words each) that the company should be using to improve their SEO
      """.format(meta_tags=meta_tags, heading_tags=heading_tags, top10keywords=top10keywords, table_string=table_string)



    def analyse_SEO(SEO_prompt):
      response = openai.Completion.create(
      model="text-davinci-003",
      prompt = SEO_prompt,
      temperature=0.7,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
      gpt3_response = response.get('choices')[0].text
      return gpt3_response,response


    seo_analysis = analyse_SEO(SEO_prompt)

    return topmetatags, topheadingtags, top10keywords, cluster_table.to_html(), cluster_plot, keyword_plot, seo_analysis[0]




# Wrap the analyze_website function with the Celery app.task decorator
@app.task
def analyze_website_task(competitor_url: str, full_site_scrape: bool = False):    
    if not full_site_scrape:
        topmetatags, topheadingtags, top10keywords, cluster_table, cluster_plot, keyword_plot, seo_analysis = analyze_single_page(competitor_url)
        return topmetatags, topheadingtags, top10keywords, cluster_table, cluster_plot, keyword_plot, seo_analysis

    sanitized_url = sanitize_url(competitor_url)
    internal_links = get_internal_links(sanitized_url)
    soup_collection = BeautifulSoup("<html><head></head><body></body></html>", "html.parser")
    
    for link in internal_links:
        try:
            soup = get_page_content(link)
            append_unique_elements(soup.head, soup_collection.head)
            append_unique_elements(soup.body, soup_collection.body)
        except Exception as e:
            print(f"Failed to analyze link: {link}. Error: {e}")

    print('got all the links')

    # Scrape and analyze meta tags
    meta_tags = get_meta_tags(soup_collection)
    topmetatags = ""
    for name, content in meta_tags.items():
        if "description" in name.lower():
            topmetatags += (f"{name}: {content}\n")

    print('fetched metatags')

    # Scrape and analyze heading tags
    heading_tags = get_heading_tags(soup_collection)
    topheadingtags = ""
    for tag, headings in heading_tags.items():
        filtered_headings = [heading for heading in headings if len(heading) > 2]
        if filtered_headings:
            topheadingtags += (f"{tag}: {', '.join(filtered_headings)}\n")

    print("fetched heading tags")

    # Scrape, analyze, and visualize keywords from page content
    page_text = soup_collection.get_text()
    page_text_cleaned = clean_text(page_text)
    preprocessed_text = preprocess_text(page_text_cleaned)

    keywords_counter = Counter(preprocessed_text)
    top10keywords = ""

    for keyword, count in analyze_keywords(keywords_counter, top_n=10):
        top10keywords += (f"{keyword}: {count}\n")

    print("fetched keywords")

    # Semantic clustering and visualization
    sentences = [preprocessed_text[i:i+10] for i in range(0, len(preprocessed_text), 10)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    words = [word for word, _ in analyze_keywords(keywords_counter, top_n=50)]
    clusters = [model.wv.doesnt_match(words)] * len(words)

    print("calculated clusters")

    cluster_plot,clusters = visualize_clusters_plot(words, model)
    cluster_table = create_cluster_table(words, model, clusters)
    keyword_plot = visualize_keywords(keywords_counter, top_n=10)


    print("plotted figures")

    table_string = cluster_table.to_string(index=False)

    print("created table string")

    heading_tags_compressed = {}

    for key, values in heading_tags.items():
        count = Counter(values)
        sorted_values = sorted(count.keys(), key=lambda x: count[x], reverse=True)
        filtered_values = [value for value in sorted_values if value.strip() != ""]
        heading_tags_compressed[key] = filtered_values[:10]


    heading_tags_clean = {}

    for key, values in heading_tags.items():
        count = Counter(values)
        sorted_values_clean = sorted(count.keys(), key=lambda x: count[x], reverse=True)
        heading_tags_clean = [value for value in sorted_values_clean if value.strip() != ""]
    
    print("cleaned up heading tags")


    SEO_prompt = f"""The following information is given about a company's website:

      Meta Tags:
      {{meta_tags}}

      Heading Tags:
      {{heading_tags_compressed}}

      Top 10 Keywords:
      {{top10keywords}}

      The following table represents clusters of thematically related words identified using NLP and clustering techniques. Each column represents a different cluster, and the words in each column are thematically related.

      {table_string}

      Please analyze the provided information and perform the following tasks:
      1. Predict what the website is all about (the market sector).
      2. Based on the market sector of the company, give a name to each cluster based on the theme it represents. The name needs to be the best summary of all the words in the cluster.
      3. Perform a SWOT analysis (Strengths, Weaknesses, Opportunities, and Threats) from an SEO perspective for the company as a whole, taking into account the meta tags, heading tags, top 10 keywords, and the clusters.
      Please provide your analysis in a clear and concise manner.
      4. Lastly, suggest a list of 10 words and 10 phrases that the company should be using to improve their SEO
      """.format(meta_tags=meta_tags, heading_tags_compressed=heading_tags_compressed, top10keywords=top10keywords, table_string=table_string)

    print("defined SEO prompt")

    def analyse_SEO(SEO_prompt):
      response = openai.Completion.create(
      model="text-davinci-003",
      prompt = SEO_prompt,
      temperature=0.7,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
      )
      gpt3_response = response.get('choices')[0].text
      return gpt3_response,response


    seo_analysis = analyse_SEO(SEO_prompt)

    print("ran seo analysis")

    print(topmetatags, heading_tags_clean,top10keywords,cluster_table.to_html(), cluster_plot, keyword_plot,seo_analysis[0])


    return topmetatags, heading_tags_clean, top10keywords, cluster_table.to_html(), cluster_plot, keyword_plot, seo_analysis[0]



gr.Interface(
    fn=analyze_website_task,
    inputs=[competitor_url_input, full_site_scrape_checkbox],
    outputs=[
        meta_tags_output,
        heading_tags_output,
        top10keywords_output,
        cluster_table_output,
        cluster_plot_output,
        keyword_plot_output,
        seo_analysis_output,
    ],
    title="SEO Analysis Tool",
    description="Enter a competitor URL to perform a SEO analysis (some javascript pages will deny full scrape).",
    layout="vertical"
).launch(share=True,debug=True)