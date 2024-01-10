"""Lamnda function to send daily summary emails."""

from os import environ
from datetime import date
import json
import logging
from dotenv import load_dotenv
import pandas as pd
import boto3
import psycopg2
import openai


DATE_TODAY = date.today().strftime("%A %B %d %Y")


def handle_openai_errors(err):
    """OpenAI API request error-handling as per official docs."""
    if isinstance(err, openai.APIError):
        logging.exception("OpenAI API returned an API Error: %s", err)
    elif isinstance(err, openai.APIConnectionError):
        logging.exception("Failed to connect to OpenAI API: %s", err)
    elif isinstance(err, openai.RateLimitError):
        logging.exception("OpenAI API request exceeded rate limit: %s", err)
    else:
        logging.exception("Unexpected error: %s", err)
    raise err


def get_db_connection():
    """Returns a database connection."""
    try:
        return psycopg2.connect(
            host=environ["DB_HOST"],
            port=environ["DB_PORT"],
            database=environ["DB_NAME"],
            user=environ["DB_USER"],
            password=environ["DB_PASSWORD"]
            )
    except psycopg2.OperationalError as error:
        logging.exception("Error connecting to the database: %s", error)
        raise


def load_stories_data() -> pd.DataFrame:
    """Loads stories with greatest score change over last 24hrs from RDS.
    Returns them as a Dataframe object."""
    query = """
            SELECT
                records.story_id,
                MAX(score) - MIN(score) AS score_change,
                stories.title,
                stories.story_url,
                MAX(record_time) AS latest_update
            FROM records
            JOIN stories ON records.story_id = stories.story_id
            WHERE record_time >= NOW() - INTERVAL '24 hours'
            GROUP BY records.story_id, stories.title, stories.story_url
            ORDER BY score_change
                DESC LIMIT 5
            ;
            """
    return pd.read_sql(query, con=get_db_connection())


def get_url_list(dataframe: pd.DataFrame) -> list:
    """Gets a list of story URLs from a dataframe."""
    return dataframe['story_url'].to_list()


def summarise_stories(url_list:list[str]) -> str: # not sure the type of the output here
    """Uses the OpenAI API to generate summaries for a list of URLs."""
    system_content_spec = """You are a newsletter writer, producing a newsletter
                        similar to https://www.morningbrew.com/daily/issues/latest."""
    user_content_spec = f"""Write a summary approximately 200 words in length,
                        that gives key insights for articles in list: {url_list},
                        return a list of dictionaries with keys 'article_title' which
                        includes the name of the article and 'summary for each article'."""

    client = openai.OpenAI(api_key=environ["OPENAI_API_KEY"])
    try:
        response = client.chat.completions.create(
            model=environ["GPT-MODEL"],
            messages=[
                {"role": "system", "content": system_content_spec},
                {"role": "user", "content": user_content_spec}],
            temperature=1
            )
        return response.choices[0].message.content.strip()
    except openai.APIError as error:
        return handle_openai_errors(error)


def generate_summaries_dict() -> dict:
    """Creates a dictionary containing all"""
    top_stories = load_stories_data()
    top_url_list = get_url_list(top_stories)
    summaries = summarise_stories(top_url_list)
    return json.loads(f"{summaries}")


def make_article_box_html(article: dict) -> str:
    """Takes in an article and returns a html string encapsulating the details of each article."""
    return f"""<body style="border-width:3px;
                     border-style:solid; border-color:#E6E6FA;
                     border-radius: 12px;
                     padding: 20px;
                     border-spacing: 10px 2em;">
       <h2 style="color: #008B8B;">{article.get('article_title')}</h2>
       <p style="color:#6495ED">{article.get('summary')}</p> </body>"""


# def generate_html_string(summaries_dict: dict) -> str:
#     """Generates a html string for the contents of an email."""
#     html_start = """<html>
#                         <head>
#                         </head>
#                         <body>
#                         <center class="wrapper">
#                             <table class="main" width="700">
#                             <tr>
#                                 <td height="8" style="background-color: #F0F8FF;">
#                                 </td>
#                                 </tr>
#                         <h1> Daily Brief</h1>
#                         <h1 style="color:#5F9EA0">Top Stories</h1>"""
#     html_end = """</body>
#                     </table>
#                 </center>
#                 </body>
#                 </html>"""

#     # --- DO NOT DELETE ---
#     # ORIGINAL WAY TO MAKE ARTICLE_LIST
#     #
#     #
#     # articles_list = []
#     # for article in summaries_dict:
#     #     article_box = f"""<body style="border-width:3px;
#     #                         border-style:solid; border-color:#E6E6FA;
#     #                         border-radius: 12px;
#     #                         padding: 20px;
#     #                         border-spacing: 10px 2em;">
#     #                     <h2 style="color: #008B8B;"> {article.get('article_title')}</h2>
#     #                     <p style="color:#6495ED"> {article.get('summary')} </p> </body>"""
#     #     articles_list.append(article_box)

#     # --- NEW WAY WE SHOULD TRY ---
#     articles_list = [make_article_box_html(article) for article in summaries_dict]

#     articles_string = ' '.join(articles_list)
#     html_full = html_start + articles_string + html_end

#     return html_full



def generate_html_string() -> str:
    '''Generates HTML string for the email.'''
    url_list = get_url_list()
    summary = summarise_story(url_list)
    dict_of_summary = json.loads(f"{summary}")
    html_start = f"""<html>
    <head>
    </head>
    <body>
    <center class="wrapper">
        <table class="main" width="700">
        <tr>
            <td height="8" style="background-color: #F0F8FF;">
            </td>
            </tr>
    <h1> Daily Brief</h1>
    <h1 style="color:#5F9EA0">Top Stories</h1>"""

    html_end="""</body>
        </table>
    </center>
    </body>
    </html>"""
    articles_list = []
    for article in dict_of_summary:
        title = article.get('article_title')
        summary = article.get('summary')
        creation_date = article.get('creation_date')
        story_url = article.get('story_url')
        author = article.get('author')

        article_box = f"""<body style="border-width:3px; border-style:solid; border-color:#E6E6FA; border-radius: 12px; padding: 20px; border-spacing: 10px 2em;">
        <h2 style="color: #008B8B;"> {title}</h2>
        <p style="color:#6495ED"> {summary} </p>
        <div>
        <p style="margin-bottom:0;">
            <a hred={story_url}> Read Article </a> |
            <p> "{creation_date}" </p> |
            <p> "{author}" </p>
            </div>
            </body>"""
        articles_list.append(article_box)
    articles_string = " ".join(articles_list)
    html_full = html_start + articles_string + html_end
    return html_full


def send_email(html_string: str):
    """Sends email newsletter using generated html string."""

    client = boto3.client('ses',
                          region_name='eu-west-2',
                          aws_access_key_id=environ["ACCESS_KEY_ID"],
                          aws_secret_access_key=environ["SECRET_ACCESS_KEY"])

    response = client.send_email(
        Destination={
                'ToAddresses': ['trainee.anurag.kaur@sigmalabs.co.uk',
                                # 'trainee.kevin.chirayil@sigmalabs.co.uk',
                                # 'trainee.kayode.apena@sigmalabs.co.uk',
                                # 'trainee.jack.hayden@sigmalabs.co.uk',
                                # more?
                                ]
                # might need everyone added as a BccAddress instead (see docs)
            },
        Message={
                'Body': {
                    'Html': {
                        'Charset': 'UTF-8',
                        'Data': html_string
                    }
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': f'Daily Brief {DATE_TODAY}',
                },
            },
        Source='trainee.anurag.kaur@sigmalabs.co.uk'
    )

    return response


def handler(): #event=None, context=None
    """Handler function."""
    load_dotenv()
    summaries_data = generate_summaries_dict()
    html_str = generate_html_string(summaries_data)
    return send_email(html_str)
