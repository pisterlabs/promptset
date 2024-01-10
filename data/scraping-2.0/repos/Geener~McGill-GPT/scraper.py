# import langchain.document_loaders import ReadTheDocsLoader
import requests
from bs4 import BeautifulSoup
import csv
import os
from urllib.parse import urlparse, urljoin


def scrape_McGill():
    base_url = 'https://www.mcgill.ca/'

    # Initialize the set of visited URLs and add the base URL
    visited_urls = set()
    visited_urls.add(base_url)

    # Initialize the set of URLs to be visited and add the base URL
    urls_to_visit = set()
    urls_to_visit.add(base_url)

    # Parse the base URL to get the domain name
    base_domain = urlparse(base_url).netloc

    # Initialize the CSV writer
    csv_file = open('links.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)


    # Loop through the URLs to be visited
    while urls_to_visit:

        # Get the next URL from the set of URLs to be visited
        current_url = urls_to_visit.pop()

        try:
            # Make a GET request to the URL
            response = requests.get(current_url)

            # Check if the response was successful (status code 200)
            if response.status_code == 200:

                # Parse the HTML content of the page
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all the links on the page
                for link in soup.find_all('a'):

                    # Get the URL from the link
                    link_url = link.get('href')

                    # Make sure the link URL is not None and is not an empty string
                    if link_url is not None and link_url != '':
                        print(link_url)
                        # Normalize the link URL by joining it with the base URL
                        link_url = urljoin(current_url, link_url)

                        # Parse the domain name from the link URL
                        link_domain = urlparse(link_url).netloc

                        # Check if the link domain matches the base domain
                        if link_domain == base_domain:

                            # Add the link URL to the set of URLs to be visited if it hasn't been visited yet
                            if link_url not in visited_urls:
                                urls_to_visit.add(link_url)

                            # Add the link URL to the set of visited URLs
                            visited_urls.add(link_url)

                            # Write the link URL to the CSV file
                            csv_writer.writerow([link_url])

            # Print an error message if the response was not successful
            else:
                print('Error: ' + str(response.status_code))

        # Print an error message if there was an error making the GET request
        except requests.exceptions.RequestException as e:
            print('Error: ' + str(e))

    # Close the CSV file
    csv_file.close()



def check_duplicates():
    url_set = set()
    no_of_duplicates = 0

    with open('links.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        print(f"no of urls found {len(list(reader))}")

        for row in reader:
            url = row[0].strip()

            if url in url_set:
                no_of_duplicates += 1
                print(f"Duplicate URL found: {url}")

            else:
                url_set.add(url)

    print(f"No of duplicates found {no_of_duplicates}")
    print("Duplicate check complete!")

def extract_data():
    # Create the data_output directory if it does not already exist
    if not os.path.exists('output'):
        os.makedirs('output')

    counter = 0

    with open('links.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            url = row[0]
            response = requests.get(url)

            # Check if response object is not None
            if response is not None:
                soup = BeautifulSoup(response.content, 'html.parser')
                main_tag = soup.find('div', id='container')

                if main_tag is not None:
                    main_text = main_tag.get_text(separator='\n', strip=True)

                    # Create a filename based on the url
                    filename = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '') + '.txt'

                    # Save the text to a file in the data_output directory
                    if "?" not in filename:
                        with open('output/' + filename, 'w', encoding="utf-8") as f:
                            print(counter)
                            counter += 1
                            f.write(main_text)
                else:
                    print(f"No main tag found in {url}")
            else:
                print(f"Error fetching {url}")

if __name__ == '__main__':
    # scrape_McGill()
    # check_duplicates()
    extract_data()