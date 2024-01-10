import argparse
import json
import requests
import os
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import openai
import re
import time
import xml.etree.ElementTree as ET
import webbrowser
import subprocess
import http.server
import socketserver
import threading
import openai_util
import openai.error

FILTER_THRESHOLD = 0.65
NUMBER_OF_PAGES = 20

httpd = None

# Replace with your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

def scrape_hackernews_page(page_number):
    url = f"https://news.ycombinator.com/?p={page_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr", class_="athing")

    articles = []
    for row in rows:
        titleline = row.find("span", class_="titleline")
        a_tag = titleline.find("a")
        title = a_tag.text
        url = a_tag["href"]
        articles.append({"title": title, "url": url})

    return articles

def sanitize_api_response(response_text):
    response_text = response_text.strip()
    json_objects = re.findall(r'\{[^}]*\}', response_text)
    valid_json_objects = []

    for json_object in json_objects:
        try:
            valid_json_objects.append(json.loads(json_object))
        except json.JSONDecodeError:
            pass

    return valid_json_objects

def load_failed_articles(directory):
    failed_file = os.path.join(directory, "failed.json")
    if os.path.exists(failed_file):
        with open(failed_file, "r") as f:
            return json.load(f)
    else:
        return []

def save_failed_articles(directory, failed_articles):
    with open(f"{directory}/failed.json", "w") as f:
        json.dump(failed_articles, f)

def create_rss_feed(feed_path, articles, category):
    rss = ET.Element("rss", {"xmlns:g": "http://base.google.com/ns/1.0", "version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Filtered Hacker News Articles"
    ET.SubElement(channel, "link").text = "https://news.ycombinator.com/"

    for article in articles:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = article["title"]
        ET.SubElement(item, "link").text = article["url"]
        ET.SubElement(item, "category").text = category
        if "summary" in article:
            ET.SubElement(item, "description").text = article["summary"]

    rss_tree = ET.ElementTree(rss)
    with open(feed_path, "wb") as f:
        rss_tree.write(f, encoding="utf-8", xml_declaration=True)

def filter_articles_using_similarity(articles, run_directory):
    tag_sets = [
        [
            "retrocomputing",
            "vintage computing",
            "80s computers",
            "8bit computer",
        ],
        [
            "nintendo",
            "nes",
            "snes",
            "gameboy",
        ],
        [
            "basic",
            "assembly",
            "forth",
        ]
    ]

    threshold = FILTER_THRESHOLD
    filtered_articles = []
    failed_articles = load_failed_articles(run_directory)
    matches = [0] * len(tag_sets)

    for article in articles:
        for i, tags in enumerate(tag_sets):
            tag_sentence = '[' + ', '.join([f'"{tag}"' for tag in tags]) + ']'
            title = article["title"]
            prompt = f"Given the following topics: {tag_sentence}. Please rate the relevance of the article \"{title}\" to these topics on a scale from 0 to 1. Just return the rating – no text. If you can not rate, return \"0.0\""

            try: 
              print('Sending Request...')
              response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=10, n=1, stop=None, temperature=0.5)

              # Extract the float value from the response
              score_text = response.choices[0].text.strip()
              if "article is not relevant" in score_text:
                  continue
              else:
                  match = re.search(r"[-+]?\d*\.\d+|\d+", score_text)
                  if match:
                      score = float(match.group())
                      if score >= threshold:
                          article["score"] = score
                          filtered_articles.append(article)
                          matches[i] += 1
                          break
                  else:
                      score = 0.0
                      print(f"Warning: No score found for article '{title}'. Adding this article to 'failed.json'. Score text was: {score_text}")
                      if not any(failed_article["url"] == article["url"] for failed_article in failed_articles):
                          failed_articles.append(article)
                      continue

            except openai.error.RateLimitError as e:
                retry_time = e.retry_after if hasattr(e, 'retry_after') else 10
                print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                return filter_articles_using_similarity(articles, run_directory)

            except openai.error.APIError as e:
                retry_time = e.retry_after if hasattr(e, 'retry_after') else 10
                print(f"API error occurred. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                return filter_articles_using_similarity(articles, run_directory)
         
            except OSError as e:
                retry_time = 5  # Adjust the retry time as needed
                print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")      
                time.sleep(retry_time)
                return filter_articles_using_similarity(articles, run_directory)

    # Save the updated failed articles
    save_failed_articles(run_directory, failed_articles)

    return filtered_articles, score

def find_newest_run_directory():
    data_dir = "data"
    run_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("run_")]
    run_dirs.sort(reverse=True)
    return os.path.join(data_dir, run_dirs[0]) if run_dirs else None

def load_results(run_directory):
    results_path = os.path.join(run_directory, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return []

def save_results(directory, results):
    with open(f"{directory}/results.json", "w") as f:
        json.dump(results, f)

    # Create RSS feeds for both successful and failed articles
    create_rss_feed(f"{directory}/results.rss", results, "results")
    failed_articles = load_failed_articles(directory)
    create_rss_feed(f"{directory}/failed.rss", failed_articles, "unscored")

def extract_content_from_url(url):
    try:
        response = requests.get(url)
    except requests.exceptions.MissingSchema:
        print(f"Error: Invalid URL '{url}'. Skipping this URL.")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")

    body = soup.find("body") or soup.find("main")
    if body:
        content = ' '.join([p.get_text() for p in body.find_all("p")])
        return content
    return ""

def update_rss_with_summaries(run_directory, results):
    rss_path = os.path.join(run_directory, "results.rss")
    if not os.path.exists(rss_path):
        print("RSS file not found. Skipping RSS update.")
        return

    # Load the existing RSS file
    with open(rss_path, "r") as f:
        rss_data = f.read()

    # Parse the RSS data
    root = ET.fromstring(rss_data)

    for entry in results:
        url = entry["url"]
        if "summary" in entry:
            summary = entry["summary"]

            # Find the matching item in the RSS feed
            for item in root.iter("item"):
                if item.find("link").text == url:
                    # Update the description element with the new summary
                    description = item.find("description")
                    if description is None:
                        description = ET.SubElement(item, "description")
                    description.text = summary
                    break

    # Save the updated RSS file
    updated_rss_data = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    with open(rss_path, "w") as f:
        f.write(updated_rss_data)

def summarize_content(content):
    max_tokens = 1500
    max_prompt_tokens = 4096 - max_tokens  # Maximum tokens allowed for prompt

    # Ensure the content is within the allowed token limit
    content_truncated = openai_util.truncate_text_to_token_limit(content, max_prompt_tokens)

    prompt = f"Please summarize the following content in no more than 10 sentences:\n\n{content_truncated}\n\nSummary:"
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5
        )
        summary = response.choices[0].text.strip()
    except openai.error.InvalidRequestError:
        summary = "Summary generation failed"
    return summary

def summarize_articles(run_directory):
    results_file = os.path.join(run_directory, "results.json")

    # Load existing results
    with open(results_file, "r") as f:
        results = json.load(f)

    # Iterate through the articles and summarize their content
    for entry in results:
        url = entry["url"]
        if "summary" not in entry:
            content = extract_content_from_url(url)
            if content:
                summary = summarize_content(content)
                entry["summary"] = summary
                print(f"Summary for {url}: {summary}")
            else:
                print(f"Unable to extract content from {url}")
            time.sleep(2)  # To avoid making too many requests in a short period of time

    # Save the updated results with summaries
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Update the RSS file with the new summaries
    update_rss_with_summaries(run_directory, results)

def serve_rss(run_directory):
    global httpd

    class RSSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                # Load the results RSS
                results_rss_path = os.path.join(run_directory, "results.rss")
                with open(results_rss_path, "r", encoding="utf-8") as f:
                    results_rss_tree = ET.parse(f)
                    results_rss_channel = results_rss_tree.find("channel")

                # Load the failed RSS
                failed_rss_path = os.path.join(run_directory, "failed.rss")
                with open(failed_rss_path, "r", encoding="utf-8") as f:
                    failed_rss_tree = ET.parse(f)
                    failed_rss_channel = failed_rss_tree.find("channel")

                # Create a new RSS feed that combines the results and failed feeds
                combined_rss = ET.Element("rss", {"version": "2.0"})
                combined_rss.append(results_rss_channel)

                for item in failed_rss_channel.findall("item"):
                    combined_rss_channel = combined_rss.find("channel")
                    combined_rss_channel.append(item)

                # Stream the combined RSS to the client
                self.send_response(200)
                self.send_header("Content-type", "application/rss+xml")
                self.end_headers()
                self.wfile.write(ET.tostring(combined_rss, encoding="utf-8"))
            else:
                self.send_error(404, "File not found")

    port = 4000
    handler = RSSRequestHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving RSS on port {port}")
        httpd.serve_forever()

def shutdown_server():
    global httpd
    if httpd:
        print("Shutting down previous server instance...")
        httpd.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Scrape and filter Hacker News articles.")
    parser.add_argument("-n", "--new", action="store_true", help="Create a new run directory.")
    parser.add_argument("-s", "--summarize", action="store_true", help="Summarize the articles in the newest run directory.")
    parser.add_argument("-o", "--open", action="store_true", help="Open the RSS feed in the newest run directory.")
    parser.add_argument("-m", "--markdown", action="store_true", help="Generate Markdown file with articles in the newest run directory.")
    args = parser.parse_args()

    if args.open:
        run_directory = find_newest_run_directory()
        if run_directory:
            rss_path = os.path.join(run_directory, "results.rss")
            if os.path.exists(rss_path):
                shutdown_server()
                t = threading.Thread(target=serve_rss, args=(run_directory,))
                t.start()
                webbrowser.get("open -a /Applications/Google\ Chrome.app %s").open(f"http://localhost:4000")
                return
            else:
                print("RSS file not found. Make sure to run with -s option first to generate the RSS file.")
                return
        else:
            print("No run directories found.")
            return

    if args.summarize:
        run_directory = find_newest_run_directory()
        if run_directory:
            summarize_articles(run_directory)
        else:
            print("No run directories found.")
        return

    if args.markdown:
        run_directory = find_newest_run_directory()
        if run_directory:
            results_path = os.path.join(run_directory, "results.json")
            failed_path = os.path.join(run_directory, "failed.json")
            articles = []
            with open(results_path, "r") as f:
                articles.extend(json.load(f))
            with open(failed_path, "r") as f:
                articles.extend(json.load(f))
            articles.sort(key=lambda article: article["score"] if "score" in article else 0, reverse=True)
            md_file_path = os.path.join(run_directory, "8bitnews.md")
            with open(md_file_path, "w") as f:
                f.write("## Intro\n\nHello 8bit'ers,\n\n## News\n\n")
                f.write("## Learn\n\n## Fun\n\n## Outro\n\n\n")
                for article in articles:
                    title = article["title"]
                    url = article["url"]
                    try:
                        description = article["summary"]
                    except KeyError:
                        description = ""
                    # description = article["summary"]
                    tags = article.get("tags", ["8bitnews", "retrocomputing", "retrogaming", "8bit"])
                    f.write(f"### {title}\n{url}\n\n{description}\n\n**_Social_**\n\nTitle\n\nLink\n\n#8bitnews #retrocomputing #retrogaming #8bit\n\n___\n")
            print(f"Markdown file created at {md_file_path}")
        else:
            print("No run directories found.")
        return

    new_run = args.new

    if new_run:
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_directory = f"data/run_{run_timestamp}"
        os.makedirs(run_directory, exist_ok=True)
        known_articles = []
    else:
        run_directory = find_newest_run_directory()
        if not run_directory:
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_directory = f"data/run_{run_timestamp}"
            os.makedirs(run_directory, exist_ok=True)
            known_articles = []
        else:
            known_articles_path = os.path.join(run_directory, "known.json")
            if os.path.exists(known_articles_path):
                with open(known_articles_path, "r") as f:
                    known_articles = json.load(f)
            else:
                known_articles = []

    scrape_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scrape_directory = f"{run_directory}/scrape_{scrape_timestamp}"
    os.makedirs(scrape_directory, exist_ok=True)

    all_articles = []
    existing_results = load_results(run_directory)

    for i in range(1, NUMBER_OF_PAGES + 1):
        articles = scrape_hackernews_page(i)
        with open(f"{scrape_directory}/articles_{i}.json", "w") as f:
            json.dump(articles, f)

        filtered_articles = []
        for article in articles:
            if not any(result["url"] == article["url"] for result in existing_results):
                if not any(known_article["url"] == article["url"] for known_article in known_articles):
                    filtered_article, score = filter_articles_using_similarity([article], run_directory)
                    if filtered_article:
                        filtered_articles.extend(filtered_article)
                    known_articles.append({"title": article["title"], "url": article["url"], "score": score})

        all_articles.extend(filtered_articles)

    unique_articles = []
    for article in all_articles:
        if not any(result["url"] == article["url"] for result in existing_results):
            unique_articles.append(article)

    unique_articles.sort(key=lambda article: article["score"], reverse=True)

    # Merge existing_results and unique_articles, and sort the combined list by
    combined_results = sorted(existing_results + unique_articles, key=lambda article: article["score"], reverse=True)

    known_articles_path = os.path.join(run_directory, "known.json")
    with open(known_articles_path, "w") as f:
        json.dump(known_articles, f)

    save_results(run_directory, existing_results + unique_articles)
    print(f"Scraping and filtering complete. Check the '{run_directory}' directory for results.")

if __name__ == "__main__":
    main()
