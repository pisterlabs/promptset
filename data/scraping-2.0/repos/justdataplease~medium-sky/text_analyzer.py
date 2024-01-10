import re
from collections import Counter
import datetime
from urllib.parse import urlparse
from excluded_urls import EXCLUDE_URLS

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.util import ngrams
import validators
import os
import csv
import openai


def tag_visible(element) -> bool:
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    return True


def get_ngrams(words: list, n: int) -> list:
    n_grams = ngrams(words, n)
    return [' '.join(grams) for grams in n_grams]


def pos_tagger(words: list) -> dict:
    # Part-of-speech tag each token
    pos_tags = pos_tag(words)

    # Count the number of adjectives, nouns, and verbs
    num_adjectives = len([word for word, pos in pos_tags if pos in ['JJ', 'JJR', 'JJS']])
    num_nouns = len([word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']])
    num_verbs = len([word for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])

    return {"adj": num_adjectives, "noun": num_nouns, "verb": num_verbs}


def get_day_of_week(date):
    day_of_week = date.strftime("%A")
    return day_of_week


def text_stemmer(words: list) -> dict:
    stemmer = PorterStemmer()

    # Use the PorterStemmer to stem each word, excluding stopwords
    stop_words = set(stopwords.words('english'))

    # Map a stem word with a normal word (randomly)
    stemmed_normal_index = {stemmer.stem(word): word for word in words if word not in stop_words}

    # Exclude english stop words and words with length <=2
    stemmed_words = [stemmed_normal_index[stemmer.stem(word)] for word in words if word not in stop_words and len(word) > 2]
    return stemmed_words


def count_external_domains(articles: dict) -> int:
    """
    To calculate external domains for each profile we create a list with the unique external domains for each article, for all articles
    """
    domains = []
    for links_per_article in articles:
        domains_per_article = []
        for link in links_per_article["links"]:
            href = link[1]
            if validators.url(href) and not re.search(EXCLUDE_URLS + "|medium", href):
                domains_per_article.append(urlparse(href).netloc)
        domains.extend(list(set(domains_per_article)))
    return len(domains)


def counts(words: list, include_stemming=True) -> dict:
    """
    Calculates article statistics : most common words, most common bigrams/trigrams etc
    """
    if include_stemming:
        # Create a PorterStemmer object
        stemmed_words = text_stemmer(words)
    else:
        stemmed_words = words

    # Count the frequency of each stemmed word
    word_counts = Counter(stemmed_words)

    # Find most frequent words
    most_common_words = word_counts.most_common(30)

    # Create a list of bigrams and count their frequency
    bigrams = get_ngrams(stemmed_words, 2)
    bigram_counts = Counter(bigrams)

    # Find most frequent bigrams
    most_common_bigrams = bigram_counts.most_common(15)

    # Create a list of trigrams and count their frequency
    trigrams = get_ngrams(stemmed_words, 3)
    trigram_counts = Counter(trigrams)

    # Find most frequent trigrams
    most_common_trigrams = trigram_counts.most_common(10)

    # Get article type
    if len(stemmed_words) < 100:
        words_num_cat = "short"
    elif len(stemmed_words) < 500:
        words_num_cat = "normal"
    elif len(stemmed_words) < 1000:
        words_num_cat = "medium"
    elif len(stemmed_words) < 1800:
        words_num_cat = "large"
    elif len(stemmed_words) > 1800:
        words_num_cat = "very large"

    return {"words": stemmed_words, "words_all": words, "word_counts": word_counts, "most_common_words": most_common_words,
            "bigrams": bigrams, "bigram_counts": bigram_counts, "most_common_bigrams": most_common_bigrams,
            "trigrams": bigrams, "trigram_counts": bigram_counts, "most_common_trigrams": most_common_trigrams,
            "words_num_all": len(words), "words_num": len(stemmed_words), "words_num_cat": words_num_cat,
            "unique_words_num_all": len(list(set(words))), "unique_words_num": len(list(set(stemmed_words))),
            }


def html_to_words(soup):
    # Find all text content in the HTML document
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    text = u" ".join(t.strip() for t in visible_texts)

    # Split the text content into words
    words = re.findall('\w+', text.lower())
    return words


def page_analyzer(soup) -> dict:
    # Parse the HTML content using BeautifulSoup
    try:
        h1 = soup.find('h1').text.strip()
    except Exception as exc:
        print("no h1 found...")
        h1 = ""

    try:
        h2 = soup.find(['h2', 'h3', 'h4']).text.strip()
    except Exception as exc:
        print("no h2 found...")
        h2 = ""

    # Split the text content into words
    words = html_to_words(soup)

    pos_tags = pos_tagger(words)
    counters = counts(words)

    rs = {"h1": h1, "h2": h2}

    return {**counters, **rs, **pos_tags}


def safe_div(x: int, y: int) -> float:
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def counter_to_text(lst: list) -> str:
    return ", ".join([f"{x[0]}({x[1]})" for x in lst])


def find_list_div_avg(list1, list2):
    total = 0
    for i in range(len(list1)):
        total += safe_div(list1[i], list2[i])

    average = total / len(list1)
    return average


def find_dates_frequency(dates: list) -> float:
    # Sort the list of dates in ascending order
    dates.sort()

    # Calculate the frequency of each interval
    freq = []
    for i in range(len(dates) - 1):
        interval = dates[i + 1] - dates[i]
        freq.append(interval.days)

    # Calculate the average frequency
    avg_freq = sum(freq) / len(freq)
    min_date = str(max(dates))
    max_date = str(min(dates))

    return avg_freq, max_date, min_date


def days_between(d1: datetime, d2: datetime = None) -> int:
    if d2 is None:
        d2 = datetime.date.today()
    else:
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d").date()
    return (d2 - d1).days


def stats_to_text(article_stats: dict, article_chars: dict, user_chars: dict) -> str:
    claps_per_person = safe_div(article_chars["clap_count"], article_chars["voter_count"])
    voter_follower = safe_div(article_chars["voter_count"], user_chars["info"]["followers_count"])

    return f"""
        <b>Heading 1</b>: {article_stats["h1"]}<br>
        <b>Heading 2</b>: {article_stats["h2"]}<br>
        <b>ChatGPT Summary</b>:<br> {article_chars["chatgpt"]["summary"]}<br>
        <br>
        <b>Publication</b>: <a href='{article_chars["publisher_url"]}'>{article_chars["publisher_name"]}</a> <br>
        <b>Published At</b>: {str(article_chars["published_at"]["date"])} {article_chars["published_at"]["time_period"]}<br>
        <b>Voters - Followers %</b>: {round(voter_follower * 100, 1)}%<br>
        <b>Claps per Person</b>: {round(claps_per_person, 1)} ({article_chars["voter_count"]} / {article_chars["clap_count"]})<br>
        <b>Responses</b>: {article_chars["post_responses"]}<br>
        <br>
        <b>Word Count (All)</b>: {article_stats["words_num_all"]}<br>
        <b>Word Count (Stemmed)</b>: {article_stats["words_num"]} ({article_stats["words_num_cat"]})<br>
        <b>Stemmed words / words</b>: {round(safe_div(article_stats["words_num"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["words_num"]} / {article_stats["words_num_all"]})<br>
        <b>Unique words / words</b>: {round(safe_div(article_stats["unique_words_num_all"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["unique_words_num_all"]} / {article_stats["words_num_all"]})<br>
        <b>Unique words / words (stemmed)</b>: {round(safe_div(article_stats["unique_words_num"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["unique_words_num_all"]} / {article_stats["words_num_all"]})<br>
        <b>Verb / words</b>: {round(safe_div(article_stats["verb"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["verb"]} / {article_stats["words_num_all"]})<br>
        <b>Adj / words</b>: {round(safe_div(article_stats["adj"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["adj"]} / {article_stats["words_num_all"]})<br>
        <b>Noun / words</b>: {round(safe_div(article_stats["noun"], article_stats["words_num_all"]) * 100, 1)}% ({article_stats["noun"]} / {article_stats["words_num_all"]})<br>

        <br>
        <b>ChatGPT Keywords</b>:<br> {", ".join(article_chars["chatgpt"]["keywords"])}<br><br>
        <b>Most Common Words</b>:<br> {counter_to_text(article_stats["most_common_words"])}<br><br>
        <b>Most Common Bigrams</b>:<br> {counter_to_text(article_stats["most_common_bigrams"])}<br><br>
        <b>Most Common Trigrams</b>:<br> {counter_to_text(article_stats["most_common_trigrams"])}<br><br>
        <br>
        """


def profile_to_text(all_data: dict, profile_stats: dict, fixed_last_date: datetime = None) -> str:
    words_upa_counts = profile_stats["words_upa_counts"]
    chatgpt_words_count = profile_stats["chatgpt_words_count"]
    words_counts = profile_stats["words_counts"]
    pos_stats = profile_stats["pos_stats"]

    domains_number = count_external_domains(all_data["articles"])
    article_length_cat = Counter(profile_stats['article_length_cat']).most_common(3)
    publication_count = Counter(profile_stats['publication']).most_common(10)
    published_frequency = find_dates_frequency([x['date'] for x in profile_stats["published_at"]])
    published_time_period_count = Counter([f"{x['time_period'][0]}-{x['time_period'][1]}" for x in profile_stats["published_at"]]).most_common(10)
    published_day_of_week_count = Counter([f"{get_day_of_week(x['date'])}" for x in profile_stats["published_at"]]).most_common(10)
    followers = all_data["user"]["info"]["followers_count"]

    words_all_num = len(profile_stats["user_words_all"])
    unique_words_all_num = len(set(profile_stats["user_words_all"]))
    words_num = len(profile_stats["user_words"])
    unique_words_num = len(set(profile_stats["user_words"]))

    clap_voter_avg = find_list_div_avg(profile_stats["clap_count"], profile_stats["voter_count"])
    voter_follower_avg = find_list_div_avg(profile_stats["voter_count"], [followers] * len(profile_stats["voter_count"]))

    last_date_seen = max([x["date"] for x in profile_stats["published_at"]])
    bio = all_data["user"]["info"]["bio"]

    return f"""
        <b>BIO</b>: {bio} <br>

        <b>Articles</b>: {len(all_data["articles"])} ({len(words_counts["words"])} stemmed words) <br>
        <b>Top article</b>: <a href='{profile_stats["top_article"][0]}'>{profile_stats["top_article"][1]} ({profile_stats["top_article"][2]})</a> <br>

        <b>Publications</b>: {counter_to_text(publication_count)} <br>
        <b>Followers</b>: {followers} <br>
        
        <b>Voters - Followers % (Article AVG)</b>: {round(voter_follower_avg * 100, 1)}%<br>
        <b>Claps per Person (Article AVG)</b>: {round(clap_voter_avg, 1)}<br>
        <br>
        
        <b>Preferred Published Time</b>: {counter_to_text(published_time_period_count)} <br>
        <b>Preferred Published Day</b>: {counter_to_text(published_day_of_week_count)} <br>
        <b>Preferred Article Length (stemmed)</b>: {counter_to_text(article_length_cat)} <br>
        <b>Published Frequency (AVG)</b>: per {round(published_frequency[0], 1)} days ({published_frequency[1]}/{published_frequency[2]}) <br>
        <b>Last Seen </b>: before {days_between(last_date_seen, fixed_last_date)} days<br>

        <b>External Domains per Article </b>: {round(safe_div(domains_number, len(all_data["articles"])), 1)}<br>

        <br>
        <b>Stemmed words / words</b>: {round(safe_div(words_num, words_all_num) * 100, 1)}% ({words_num} / {words_all_num})<br>
        <b>Unique words / words</b>: {round(safe_div(unique_words_all_num, words_all_num) * 100, 1)}% ({unique_words_all_num} / {words_all_num})<br>
        <b>Unique words / words (stemmed)</b>: {round(safe_div(unique_words_num, words_num) * 100, 1)}% ({unique_words_num} / {words_num})<br>
        <b>Verb / words</b>: {round(safe_div(pos_stats["verb"], words_all_num) * 100, 1)}% ({pos_stats["verb"]} / {words_all_num})<br>
        <b>Adj / words</b>: {round(safe_div(pos_stats["adj"], words_all_num) * 100, 1)}% ({pos_stats["adj"]} / {words_all_num})<br>
        <b>Noun / words</b>: {round(safe_div(pos_stats["noun"], words_all_num) * 100, 1)}% ({pos_stats["noun"]} / {words_all_num})<br>
        <br>
        
        <b>Most Common ChatGPT Keywords (UPA)</b>:<br> {counter_to_text(chatgpt_words_count)}<br><br>

        <b>Most Common Words (UPA)</b>:<br> {counter_to_text(words_upa_counts["most_common_words"])}<br><br>
        <b>Most Common Bigrams (UPA)</b>:<br> {counter_to_text(words_upa_counts["most_common_bigrams"])}<br><br>
        <b>Most Common Trigrams (UPA)</b>:<br> {counter_to_text(words_upa_counts["most_common_trigrams"])}<br><br>
        
        <b>Most Common Words</b>:<br> {counter_to_text(words_counts["most_common_words"])}<br><br>
        <b>Most Common Bigrams</b>:<br> {counter_to_text(words_counts["most_common_bigrams"])}<br><br>
        <b>Most Common Trigrams</b>:<br> {counter_to_text(words_counts["most_common_trigrams"])}<br><br>
        """


def chatgpt_api(soup, num_keyphrases=10, dummy=False):
    if dummy:
        return "hello this is a test"

    stop_words = set(stopwords.words('english'))
    words = html_to_words(soup)
    new_words = [x for x in words if x not in stop_words and len(x) > 2]

    try:
        full_text = " ".join(new_words[0:2000])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": f"""Please provide a summary and suggest the top {num_keyphrases} keywords that best describe the important topics or themes present in the following text. Your answer should include the format: KEYWORDS=keyword_1, keyword_2, ..., keyword_{num_keyphrases} and SUMMARY=summary_text.\n\n {full_text}"""},
            ]
        )
    except Exception as exc:
        print(exc)

        full_text = " ".join(new_words[0:1000])

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": f"""Please provide a summary and suggest the top {num_keyphrases} keywords that best describe the important topics or themes present in the following text. Your answer should include the format: KEYWORDS=keyword_1, keyword_2, ..., keyword_{num_keyphrases} and SUMMARY=summary_text.\n\n {full_text}"""},
            ]
        )

    reply = response["choices"][0]["message"]["content"]

    return reply


def chatgpt_parser(username, soup, article_id):
    # Define the filename and API endpoint URL
    filename = f'data\{username}_openai_responses.csv'

    # Check if the file exists
    if os.path.exists(filename):
        # If the file exists, open it and search for the ID
        with open(filename, 'r') as f:
            found = False
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for cols in reader:
                if cols[0] == article_id:
                    print(f"id {article_id} found, using local file...")
                    # If the ID is found, use the associated response
                    response = cols[1]
                    found = True
                else:
                    # Otherwise, add the line to the list of lines
                    lines.append(cols)
            if not found:
                print(f"id {article_id} not found, using the api...")
                # If the ID is not found, use the API and add the new ID and response to the file
                response = chatgpt_api(soup)
                lines.append([article_id, response])
                with open(filename, 'a', newline='', encoding='utf8') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([article_id, response])
    else:
        # If the file does not exist, use the API and create the file with the new ID and response
        response = chatgpt_api(soup)
        with open(filename, 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([article_id, response])

    # Do something with the response
    # Find the keywords and summary in the text
    keywords = "error"
    summary = "error"

    try:
        keywords = re.search(r'KEYWORDS(?:\s+)?(?:\=|\:)(?:\s+)?([\w\,\-0-9\n\t\s]+)(?:SUMMARY)?', response).group(1).split(",")
        keywords = [x.strip().lower().replace(".", "") for x in keywords]
        unikeywords = []
        for x in keywords:
            unikeywords.extend(x.split(" "))
        unikeywords = list(set(unikeywords))
    except Exception as exc:
        pass

    try:
        summary = re.search(r'SUMMARY(?:\s+)?(?:\=|\:)(?:\s+)?([\w\,\-0-9\n\t\s\.\,\(\)\'\"]+)(?:KEYWORDS)?', response).group(1).strip()
    except Exception as exc:
        pass

    return {"keywords": keywords, "summary": summary, "unikeywords": unikeywords}
