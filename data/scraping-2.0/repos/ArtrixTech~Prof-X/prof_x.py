from scholarly import scholarly
import re
import os
from utils import *
from translator import Translator
from config import TX_SECRET_ID, TX_SECRET_KEY, IMGUR_CLIENT_ID
import openai_assisted
import json
import numpy as np
import concurrent.futures
import time
import page_template


failed_authors = []


def get_authors_from_input(input_str):
    return [author.strip() for author in input_str.split(',')]


def choose_author(author_name):
    search_query = scholarly.search_author(author_name)
    authors = list(search_query)
    global failed_authors

    if len(authors) == 0:
        print(f"No results found for {author_name}.")
        failed_authors.append((author_name, "Author Not Found"))
        return None
    if len(authors) == 1:
        return authors[0]

    print(f"Found multiple authors for {author_name}:")
    for idx, author in enumerate(authors):
        print(f"{idx + 1}. {author['name']}, {author['affiliation']}")
    print("(Input x to skip this author)")

    choice = input_with_timeout(
        "Please select the correct author by entering the number: ", 25, 'x')

    if choice == 'x' or choice == 'X':
        failed_authors.append((author_name, "User Skipped/Choose Timeout"))
        return None
    return authors[int(choice) - 1]


def generate_briefing_img(author):

    start_year, curr_year = list(author['cites_per_year'])[
        0], list(author['cites_per_year'])[-1]
    tracing_year_span = curr_year-start_year+1

    heat_map_data = np.zeros((tracing_year_span, tracing_year_span))

    for curr_pub in author['publications']:

        pub_year = int(list(curr_pub['cites_per_year'])[0])
        if pub_year < start_year:
            continue

        for year in curr_pub['cites_per_year']:
            year_passed = year-pub_year
            heat_map_data[year_passed, pub_year -
                          start_year] += curr_pub['cites_per_year'][year]

    fig = generate_heatmap(author, heat_map_data,
                           start_year, curr_year, tracing_year_span)
    return save_plot_to_imgur(fig, IMGUR_CLIENT_ID)


def generate_markdown(author):

    author = scholarly.fill(author)
    print("Author info fetched.")

    # print(author)

    h_index = int(author['hindex']) if 'hindex' in author else None
    if 'publications' in author:

        if h_index:
            author['publications'] = author['publications'][:h_index]

        def fetch_publication_info(publication):
            return scholarly.fill(publication)

        print(">> Fetching publication info...")
        last_bar_length = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, _ in enumerate(author['publications']):
                future = executor.submit(
                    fetch_publication_info, author['publications'][i])
                futures.append(future)
                time.sleep(0.05)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                author['publications'][i] = future.result()
                last_bar_length = display_progress_bar(
                    i+1, len(author['publications']))

        clear_last_line(last_bar_length)
        print("Publication info fetched.")

    markdown_data = "\n"
    briefing_section = ""
    summary_section = ""
    publication_section = ""

    # Briefing Part
    print(">> Generating briefing info...")

    filling_data = {
        'profile_picture': author['url_picture'] if 'url_picture' in author else 'https://i.imgur.com/hepj9ZS.png',
        'name': author['name'],
        'position': author['affiliation'] if 'affiliation' in author else 'Unknown Affiliation',
        'h_index': f"{author['hindex']} -> {author['hindex5y']}<sub>(5y)</sub>" if 'hindex5y' in author else author['hindex'] if 'hindex' in author else 'Unknown h-index',
        'research_interests': ', '.join(author['interests']) if 'interests' in author else 'Unknown Research Interests',
        'scholar_link': f"https://scholar.google.com/citations?user={author['scholar_id']}" if 'scholar_id' in author else 'https://scholar.google.com/',
        'homepage_link': author['homepage'] if 'homepage' in author else 'No Homepage Info', # TODO: Fix the handling of homepage
    }
    rendered = page_template.fill_template(
        page_template.briefing_template, filling_data)
    briefing_section += rendered+'\n'

    briefing_img = generate_briefing_img(author)
    briefing_section += f"## Research Heatmap\n\n"
    briefing_section += f"![image]({briefing_img})\n"
    print("Briefing info generated.")

    # 著作信息部分
    if 'publications' in author:

        publicaions = author['publications']
        publication_ai_data_prep = ""
        publication_titles = []

        for i, publication in enumerate(publicaions):
            title = publication['bib'].get('title', 'Unknown Title')
            pub_year = publication['bib'].get('pub_year', 'Unknown Year')
            citation = publication['bib'].get('citation', 'Unknown Citation')
            num_citations = publication.get('num_citations', 0)
            citedby_url = publication.get('citedby_url', '#')

            publication_ai_data_prep += f"{title},{pub_year},{num_citations}\n"
            publication_titles.append(title)

        def summary_ai():
            ai_summary, total_tokens = openai_assisted.publication_summarize(
                publication_ai_data_prep)

            if ai_summary:
                print(f"AI Summarized.")
                return ai_summary, total_tokens
            else:
                result = input_with_timeout(
                    "AI summarization failed for 3 times, still retry?(y/n): ", 10, 'n')
                if result == 'y' or result == 'Y':
                    return summary_ai()
                else:
                    return None, -1

        print(">> AI Summarizing...")
        ai_summary, total_tokens = summary_ai()
        if ai_summary is None:
            failed_authors.append((author['name'], "AI Summarization Failed"))
            return None

        ai_summary = json.loads(ai_summary)

        for subject in ai_summary:
            summary_section += f"#### {subject['subject']}\n"
            for sub_area in subject['sub_areas']:
                summary_section += f"- **{sub_area['area']}**:\n"
                summary_section += f"  {sub_area['summary']}\n\n"

        tr = Translator(TX_SECRET_ID, TX_SECRET_KEY)
        translations = tr.batch_translate(publication_titles, "zh")
        print("Publication title translated.")

        for idx, publication in enumerate(publicaions):
            title = publication['bib'].get('title', 'Unknown Title')
            pub_year = publication['bib'].get('pub_year', 'Unknown Year')
            author_info = publication['bib'].get('author', 'Unknown Author')
            citation = publication['bib'].get('citation', 'Unknown Citation')
            num_citations = publication.get('num_citations', 0)
            # citedby_url = publication.get('citedby_url', '#')
            publication_url = publication.get('pub_url', '#')

            author_info = author_info.replace(' and ', ',')
            author_list = author_info.split(',')

            author_info = ''
            for i, author_name in enumerate(author_list):
                author_name = remove_symbols(author_name)
                if author_name in author['name'] or author['name'] in author_name:
                    author_name = f'<span style="text-decoration: underline; font-style: italic; font-weight: bold;">{author_name}</span>'
                author_info += f"{author_name}<sub>{i+1}</sub>"+', '

            publication_section += f"- **<{pub_year}, {num_citations}> {title}**\n"
            publication_section += f"  - {translations[idx]}\n"
            publication_section += f"  - {author_info}\n"
            publication_section += f"  - {citation}\n"
            publication_section += f"  - [{publication_url}]({publication_url})\n\n"

    markdown_data += briefing_section

    markdown_data += "\n## Research Summary\n\n"
    markdown_data += summary_section

    markdown_data += "\n## Publications\n\n"
    markdown_data += publication_section

    markdown_data += "\n## Raw Data\n"
    markdown_data += f"```json\n{json.dumps(author, indent=2)}\n```\n"

    return markdown_data


def save_to_md_file(author_info, domain, content):
    directory = f"saved/{domain}"
    os.makedirs(directory, exist_ok=True)

    h_index = author_info.get("hindex", "-1")

    filename = f"{directory}/[{h_index}]{author_info['name']}.md"

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Saved to {filename}")


def main():
    input_str = input("Please enter the list of authors separated by commas: ")
    author_names = get_authors_from_input(input_str)

    for i, author_name in enumerate(author_names):

        print(f"Processing {author_name}:")
        author = choose_author(author_name)

        if author:
            print(f">> Author found, fetching more info...")
            md_output = generate_markdown(author)
            mail_raw = author.get('email_domain', '@no_data.com')
            if '@' in mail_raw:
                save_to_md_file(author, get_top_domain(
                    mail_raw).replace('@', ''), md_output)
            else:
                save_to_md_file(author, 'unclassified', md_output)

        if i == len(author_names)-1:
            if not len(failed_authors) == 0:
                print("Failed authors:")
                for failed_author in failed_authors:
                    print(f"{failed_author[0]}: {failed_author[1]}")
                result = input_with_timeout("Retry all? (y/n)", 10, 'n')

                if result == 'y' or result == 'Y':
                    for failed_author in failed_authors:
                        author_names.append(failed_author[0])


if __name__ == '__main__':
    main()
