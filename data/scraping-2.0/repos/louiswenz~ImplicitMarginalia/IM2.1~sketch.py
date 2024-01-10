import scholarly
import requests
from bs4 import BeautifulSoup
import openai
import nltk.data

# Set up OpenAI API key
# Replace with your OpenAI API key
openai.api_key = ""


def find_citing_articles1(citation_sentence):
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={citation_sentence}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []

    for result in soup.find_all('h3', class_='gs_rt'):
        link_element = result.find('a')
        if link_element:
            link = link_element['href']
            try:
                article_response = requests.get(link)
                if article_response.status_code == 200:
                    links.append(link)
            except requests.exceptions.RequestException:
                continue

    print(len(links))

    articles_title = []
    articles_context = []

    results = soup.find_all('div', class_='gs_ri')

    for result in results:
        title_element = result.find('h3', class_='gs_rt')
        if title_element:
            title = title_element.text.strip()
            title = title.replace("[HTML]", "").strip()

            # Extracting the citation sentence and its context
            context_paragraph = result.find('div', class_='gs_rs').text

            context_paragraph = context_paragraph.replace("\n", "").strip()
            # Checking if the citation sentence is present in the context sentences
            # if citation_sentence.lower() in context_paragraph.lower():
            articles_title.append(title)

            articles_context.append(context_paragraph)

    return articles_title, articles_context


def summarize(contexts):
    # Prepare the search query
    summaries = []
    for i in contexts:
        query = f"Summarize the following text: {i}"

        # Issue the search request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive"},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            stop=None,
            n=1,
            temperature=0.3
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
    return summaries


_sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def split_sentence(text):
    # Split text.
    sentences = _sent_detector.tokenize(text)
    # Find each sentence's offset.
    needle = 0
    triples = []
    for sent in sentences:
        start = text.find(sent, needle)
        end = start + len(sent) - 1
        needle += len(sent)
        triples.append(sent)
    # Return results
    return sentences


def find_citing_articles(citation_sentence):  # not using

    citation_sentence = strip_punctuation(citation_sentence)
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={citation_sentence}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles_title = []
    articles_context = []

    results = soup.find_all('div', class_='gs_ri')

    for result in results:
        title_element = result.find('h3', class_='gs_rt')
        if title_element:
            title = title_element.text.strip()
            title = title.replace("[HTML]", "").strip()

            # Extracting the citation sentence and its context
            context_paragraph = result.find('div', class_='gs_rs').text
            context_paragraph = context_paragraph.replace("\n", "").strip()
            # Checking if the citation sentence is present in the context sentences
            if citation_sentence.lower() in context_paragraph.lower():
                articles_title.append(title)
                articles_context.append(context_paragraph)

    return articles_title, articles_context


def summarize(contexts):  # not using
    # Prepare the search query
    summaries = []
    for i in contexts:
        query = f"Summarize the following text: {i}"

        # Issue the search request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive. Talk in first person"},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            stop=None,
            n=1,
            temperature=0.3
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
    return summaries


# Example usage
text = "I have a dream"
# print(nltk.tokenize.sent_tokenize(text, language='english'))
articles_title, articles_context = find_citing_articles1(text)
# print((articles_context))
