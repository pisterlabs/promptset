
import requests
import xmltodict
import re
from bs4 import BeautifulSoup
import os
import pandas as pd
from requests.exceptions import ConnectionError
from datetime import datetime
import variables
import json
from io import StringIO
from urllib.parse import urljoin
from report_text import openai_trans

# define a function to extract date from text
def extract_date(text):
    text_without_tags = re.sub(r"<[^>]+>", "", text)
    text_without_special_chars = re.sub(r"[^a-zA-Z0-9\s]", "", text_without_tags)
    match = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text_without_special_chars)
    if match:
          return(match.group(2) + " " + match.group(1))
    else:
          return(None)

def extract_date_cn(text):
    text_without_tags = re.sub(r"<[^>]+>", "", text)
    text_without_special_chars = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", "", text_without_tags)
    match = re.search(r"(\d{4})年(\d{1,2})月", text_without_special_chars)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        date_obj = datetime(year, month, 1)
        formatted_date = date_obj.strftime("%Y %B")
        return formatted_date
    else:
        return None
    
def find_max_date(YearMonths):
    date_objects = [datetime.strptime(date, "%Y %B") for date in YearMonths]
    max_date = max(date_objects, key=lambda x: x)
    max_date_str = max_date.strftime("%Y %B")
    return max_date_str

# define a function to get PubMed RSS results
def get_rss_results(url, label, origin):
    # Send request and get response
    response = requests.get(url)

    # Check response status code
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {label} results. Status code: {response.status_code}")

    # Parse XML results
    rss_results = xmltodict.parse(response.content)

    # Extract results
    results = []
    for item in rss_results["rss"]["channel"]["item"]:
        date = extract_date(item["title"])
        date_obj = datetime.strptime(date, "%Y %B")
        formatted_date = date_obj.strftime("%Y/%m/%d")
        results.append({
            "title": item["title"],
            "pubDate": item["pubDate"],
            "dc:date": item["dc:date"],
            "date": date_obj,
            "YearMonthDay": formatted_date,
            "YearMonth": date,
            "doi": item["dc:identifier"],
            "source": label,
            "origin": origin
        })

    # Sort by date
    results.sort(key=lambda result: result["date"], reverse=False)
    # Exclude the first 4 results
    results = results[4:]

    return results


# define a function to get china cdc weekly results

def get_cdc_results(url, label, origin):
    # Send an HTTP request to get the webpage content
    response = requests.get(url)

    # Check response status code
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {label} results. Status code: {response.status_code}")

    # Parse HTML results
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    a_tags = soup.find_all("a")
    result_list = []

    # Traverse each <a> tag, extract text and link
    for a_tag in a_tags:
        text = a_tag.text.strip()
        link = a_tag.get("href")
        if link and "doi" in link and "National Notifiable Infectious Diseases" in text:
            date = extract_date(re.sub(r"[^\w\s-]", "", text))
            date_obj = datetime.strptime(date, "%Y %B")
            formatted_date = date_obj.strftime("%Y/%m/%d")
            doi = link.split("doi/")[1]
            result_list.append({
                "title": text,
                "date": date_obj,
                "YearMonthDay": formatted_date,
                "YearMonth": date,
                "link": url + link,
                "doi": ['missing', 'missing', 'doi:' + doi],
                "source": label,
                "origin": origin
            })

    result_list = list({v['link']: v for v in result_list}.values())
    return result_list

def get_gov_results(url, form_data, label, origin):
    # Send a request and get the response
    response = requests.post(url, data=form_data)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {label} results. Status code: {response.status_code}")

    # Check if the response contains data
    if response.json()['data']['results'] == []:
        raise Exception(f"Failed to fetch {label} results. No data returned.")
    
    # Extract results
    titles = [response.json()['data']['results'][i]['source']['title'] for i in range(10)]
    links = [response.json()['data']['results'][i]['source']['urls'] for i in range(10)]

    # Traverse each <a> tag, extract text and link
    result_list = []
    for title,link in zip(titles,links):
        if title and link:
            date = extract_date_cn(title)
            date_obj = datetime.strptime(date, "%Y %B")
            formatted_date = date_obj.strftime("%Y/%m/%d")
            link = json.loads(link).get('common')
            link = urljoin(url, link)
            result_list.append({
                "title": title,
                "date": date_obj,
                "YearMonthDay": formatted_date,
                "YearMonth": date,
                "link": link,
                "doi": ['missing', 'missing', 'missing'],
                "source": label,
                "origin": origin
            })
    return result_list

# define a function to get table data from URLs
def get_table_data(url):
    # Send a request and get the response
    response = requests.get(url)
    if response.status_code != 200:
        print(url)
        raise Exception("Failed to fetch web content, status code: {}".format(response.status_code))
    else:
        print("Successfully fetched web content, urls: {}".format(url))

    # Use pandas to directly parse the table data
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    tables = soup.find_all('table')[0]

    data = []
    thead = tables.find('thead')
    if thead:
        thead_rows = thead.find_all('tr')
        for tr in thead_rows:
            data.append([th.get_text().strip() for th in tr.find_all(['td', 'th'])])

    table_body = tables.find('tbody')
    if table_body:
        rows = table_body.find_all('tr')
        for tr in rows:
            cells = tr.find_all('td')
            if cells:
                data.append([td.get_text().strip() for td in cells])

    table_data = pd.DataFrame(data)

    return table_data

def clean_table_data(data, filtered_result):
    # Clean database
    data = data.iloc[1:].copy()
    data.columns = ['Diseases', 'Cases', 'Deaths']
    data['Diseases'] = data['Diseases'].str.replace(r"[^\w\s]", "", regex=True)

    # Add various columns
    column_names = ['DOI', 'URL',
                    'Date', 'YearMonthDay', 'YearMonth',
                    'Source',
                    'Province', 'ProvinceCN', 'ADCode',
                    'Incidence', 'Mortality']
    column_values = [filtered_result['doi'][2], filtered_result['link'],
                     filtered_result["date"], filtered_result["YearMonthDay"], filtered_result["YearMonth"],
                     filtered_result["source"], 'China', '全国', '100000', -10, -10]

    for name, value in zip(column_names, column_values):
        data[name] = value
    
    # trans Diseases to DiseasesCN
    diseaseCode2Name = pd.read_csv("../../Script/WeeklyReport/variables/diseaseCode2Name.csv")
    diseaseCode2Name = dict(zip(diseaseCode2Name["Code"], diseaseCode2Name["Name"]))
    data['DiseasesCN'] = data['Diseases'].map(diseaseCode2Name)

    # Reorder the column names
    column_order = ['Date', 'YearMonthDay', 'YearMonth', 'Diseases', 'DiseasesCN', 'Cases', 'Deaths', 'Incidence', 'Mortality', 'ProvinceCN', 'Province', 'ADCode', 'DOI', 'URL', 'Source']
    table_data = data[column_order]

    return table_data

def clean_table_data_cn(data, filtered_result):
    # Clean database
    data = data.iloc[1:].copy()
    data.columns = ['DiseasesCN', 'Cases', 'Deaths']
    data = data[~data['DiseasesCN'].str.contains('合计')]
    data['DiseasesCN'] = data['DiseasesCN'].str.replace(r"[^\w\s\u4e00-\u9fa5]", "", regex=True)
    data['DiseasesCN'] = data['DiseasesCN'].str.replace(r"甲乙丙类总计", "合计", regex=True)
    
    # Add various columns
    column_names = ['DOI', 'URL',
                    'Date', 'YearMonthDay', 'YearMonth',
                    'Source',
                    'Province', 'ProvinceCN', 'ADCode',
                    'Incidence', 'Mortality']
    column_values = [filtered_result["doi"][2], filtered_result['link'],
                     filtered_result["date"], filtered_result["YearMonthDay"], filtered_result["YearMonth"],
                     filtered_result["source"],
                     'China', '全国', '100000',
                     -10, -10]
    for name, value in zip(column_names, column_values):
        data[name] = value
    
    # trans DiseasesCN to Diseases
    diseaseName2Code_df = pd.read_csv("../../Script/WeeklyReport/variables/diseaseName2Code.csv")
    diseaseName2Code = dict(zip(diseaseName2Code_df["Name"], diseaseName2Code_df["Code"]))
    data['Diseases'] = data['DiseasesCN'].map(diseaseName2Code)
    
    na_indices = data['DiseasesCN'][data['DiseasesCN'].isna()].index
    for i in na_indices:
        data.loc[i, 'Diseases'] = openai_trans(os.environ["DATA_TRANSLATE_CREATE"],
                                               os.environ["DATA_TRANSLATE_CHECK"],
                                               data.loc[i, 'DiseasesCN'],
                                               diseaseName2Code.values())
        # append the new translation to diseaseName2Code_df
        diseaseName2Code_df = diseaseName2Code_df.append({'Name': data.loc[i, 'DiseasesCN'], 'Code': data.loc[i, 'Diseases']},
                                                         ignore_index=True)
    
    # if update diseaseName2Code.csv
    if len(na_indices) > 0:
        diseaseName2Code_df.to_csv("../../Script/WeeklyReport/variables/diseaseName2Code.csv", index=False)

    # Reorder the column names
    column_order = ['Date', 'YearMonthDay', 'YearMonth', 'Diseases', 'DiseasesCN', 'Cases', 'Deaths', 'Incidence', 'Mortality', 'ProvinceCN', 'Province', 'ADCode', 'DOI', 'URL', 'Source']
    table_data = data[column_order]

    return table_data

def process_table_data(results):
    """
    Process table data from URLs and save the results to CSV files.

    Args:
        results (list): A list of dictionaries containing the extracted information.

    Raises:
        Exception: If the HTTP response status code is not 200.

    Returns:
        None
    """
    urls = [result['link'] for result in results]

    for i, url in enumerate(urls):
        # Send a request and get the response
        origin = results[i]['origin']
        data_raw = get_table_data(url)
        
        if origin == 'CN':
            data = clean_table_data_cn(data_raw,
                                       results[i])
        else:
            data = clean_table_data(data_raw,
                                    results[i])

        # Save the results for each month to a CSV file
        file_name = os.path.join("WeeklyReport/", results[i]["YearMonth"] + ".csv")
        data.to_csv(file_name, index=False, encoding="UTF-8-sig")
