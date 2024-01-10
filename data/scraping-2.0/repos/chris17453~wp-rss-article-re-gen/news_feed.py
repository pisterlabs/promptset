
import requests
from bs4 import BeautifulSoup
import feedparser
from openai import OpenAI
from datetime import datetime
import json
from core import  strip_html_tags, save_article, load_article
from wordpress import publish,publish_img,get_or_create_tags,create_or_get_feed_id
from prompt import generate_content,generate_image,generate_title,validate_content




def parse_rss_feed_and_extract_content(rss_feed_url):
    try:
        # Parse the RSS feed
        feed = feedparser.parse(rss_feed_url)

        # Initialize an empty list to store article information
        articles = []
        article_count=0
        # Loop through the entries in the feed
        for entry in feed.entries:


            print(entry.title)
            
            article = {
                'link': entry.link,
                'published': entry.published,
                'summary': entry.summary,
                'entry':entry
            }
            article=load_article(article)
            follow=None
            if follow==True:
                # Follow the link to the article's webpage and extract content
                response = requests.get(entry.link)
                # Check for HTTP status code
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract the article content (you may need to customize this part)
                    content = soup.find('div', class_='article-content')
                    if content:
                        article['content'] = content.get_text()
                    else:
                        article['content'] = 'Content not found on the page.'

                else:
                    article['content'] = f'Failed to fetch the webpage. Status code: {response.status_code}'
            print("attempt")
            validate=validate_content(entry.title)
            if validate:
                print ("VALID TOPIC")
            else:
                print ("INVALID TOPIC")
                continue

            if 'content' not in article:
                print("Content")
                content = generate_content(entry.title, entry.link)
                article['content']=content
            
            # Generate title from content text
            if 'title' not in article:
                print("Title")
                title = generate_title(entry.title)
                title= strip_html_tags(title)
                article['title']=title
    
            if 'img' not in article:
                print("Generate Image")
                img_path=generate_image(title.replace("-",' '))
                article['img']=img_path

            if 'featured_media_id' not in article:
                print("Publish Image")
                article['featured_media_id']=publish_img(img_path)

            if 'tag_ids' not in article:
                print("Publishing Tags")
                tags_dict = [tag["term"] for tag in entry.tags]
                article['tag_ids']=get_or_create_tags(tags_dict)

            if 'categories' not in article:
                print("Getting/setting Category")
                
                article['categories']=create_or_get_feed_id("Feed")

            if 'id' not in article:
                print("Publising Article")
                article_id=publish(article)
                article['id']=article_id
            
            save_article(article)
            articles.append(article)

        return articles

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []



articles=parse_rss_feed_and_extract_content("https://www.techcrunch.com/rss")
if articles:
    # Pretty-print the articles list
    print(json.dumps(articles, indent=4))
else:
    print("No articles were retrieved from the RSS feed.")
    #lobste.rs

        

