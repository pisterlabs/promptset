import xml.etree.ElementTree as ET
import csv
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure your API key is set up
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_content(prompt):
    response = openai.Completion.create(
      model="gpt-3.5-turbo-instruct",
      prompt=prompt,
      temperature=0.7,
      max_tokens=200
    )
    return response.choices[0].text.strip()

def format_authors(authors):
    formatted_authors = []
    for author in authors:
        text = author.text.strip()
        parts = text.split(', ')
        if len(parts) == 2:
            last, first = parts
            initials = ''.join([name[0] + '.' for name in first.split()])
            formatted_authors.append(f"{last}, {initials}")
        else:
            formatted_authors.append(text)
    return ', '.join(formatted_authors)


def xml_to_csv(xml_file, csv_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Create a CSV file and write the header
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["Title", "Type", "Topic", "Authors", "Year", "Source", "Punchline", "Hypothesis", "Method", "Result", "Future work", "Usage", "Quick Ref", "URL"]
        writer.writerow(header)

        # Iterate through the records in the XML file
        for record in root.find('records'):
            row = []
            
            # Title
            title = record.find('titles/title')
            row.append(title.text if title is not None else '')

            # Type
            ref_type = record.find('ref-type')
            row.append(ref_type.get('name') if ref_type is not None else '')

            # Topic (keywords)
            keywords = record.find('keywords')
            row.append(', '.join([keyword.text for keyword in keywords]) if keywords is not None else '')

            # Authors
            authors = record.find('contributors/authors')
            # Inside your loop to process each record
            authors_text = format_authors(authors) if authors is not None else ''
            authors_text = authors_text.replace('\n', '')
            row.append(authors_text)

            # Year
            year = record.find('dates/year')
            row.append(year.text if year is not None else '')

            # Source (periodical)
            source = record.find('periodical/full-title')
            row.append(source.text if source is not None else '')

            # Abstract for content generation
            abstract = record.find('abstract')
            abstract_text = abstract.text if abstract is not None else ''

            # Generate content for the specified columns
            punchline = generate_content(f"{abstract_text} Find the main takeaway? Response with only one sentence.")
            hypothesis = generate_content(f"{abstract_text} Find the hypothesis? Response within 80 words.")
            method = generate_content(f"{abstract_text} Find what method was used? Response within 80 words.")
            result = generate_content(f"{abstract_text} Find what were the results? Response within 80 words.")
            future_work = generate_content(f"{abstract_text} Find what are the future directions? Response within 80 words.")

            row.extend([punchline, hypothesis, method, result, future_work, ''])

            quick_ref = (authors_text if authors_text is not None else '') + '. (' + (year.text if year is not None and year.text is not None else '') + ') ' + (title.text if title is not None else '') + '. ' + (source.text if source is not None else '')
            row.append(quick_ref)

            # Find the URL in the XML
            url = record.find(".//urls/related-urls/url")
            url_text = url.text if url is not None else ''
            row.append(url_text)

            # Write the row to the CSV file
            writer.writerow(row)

if __name__ == "__main__":
    xml_to_csv('export.xml', 'output.csv')
