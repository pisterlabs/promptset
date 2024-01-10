from pathlib import Path
import requests
from langchain.document_loaders import BSHTMLLoader
from bs4 import BeautifulSoup

from property_finder.configuration.log_factory import logger
from property_finder.configuration.config import cfg




### .sv-details__address1--truncate > span > span


def html_loader(path: Path):
    loader = BSHTMLLoader(path)
    data = loader.load()
    return data


def find_realtor(soup):
    
    realtor_name_list = []
    for realtors in soup.find_all("h6", ("sv-details__contacts-name")):
        realtor_name_list.append(realtors.string)
    return realtor_name_list


def find_name_property(soup):
    
    property_name_list = []
    address_tags = soup.select(".sv-details__address1")
    for property in address_tags:
        # logger.info(i.text)
        property_name_list.append(property.text)
    return property_name_list

def find_address(soup):
    property_address_list = []
    address_tags = soup.select(".sv-details__address2")
    for address in address_tags:
        # logger.info(i.text)
        property_address_list.append(address.text)
    return property_address_list


def find_property_details(soup):
    
    property_details_list = []
    property_details = soup.select(".sv-details__features")
    for details in property_details:
        # logger.info(i.text)
        property_details_list.append(details.text)
    return property_details_list

def find_property_link(soup):
    
    property_links_list = []
    for a in soup.find_all('a', class_="sv-details__link"):
    #print("Found the URL:", a['href']) 
        #logger.info(a['href'])
        property_links_list.append("https://search.savills.com/" + a['href'])
    return property_links_list

def create_soup(html):
    content_html = html.content
    soup = BeautifulSoup(content_html, "html.parser")
    return soup


def zip_details(property_name, property_details, property_sizes, property_address, realtor_name, property_link):
    mapped = zip(property_name, property_details, property_sizes, property_address, realtor_name, property_link)
    return mapped

def property_size(soup):
    property_size_list = []
    div_elements = soup.find_all('div', class_='sv-property-attribute sv--size')

    # Find the span element inside the div
    for div_element in div_elements:
        span_element = div_element.find('span', class_='sv-property-attribute__value')
        information = span_element.get_text()
        #logger.info(information)
        property_size_list.append(information)
    return property_size_list



def create_markdown(input_list):
    # Convert the list to a list of lists (each item is a separate list)
    table_data = []
    
    for item in input_list:
        data = {}
        data["property name"] = item[0]
        data['property details'] = item[1]
        data['property sizes'] = item[2]
        data['property address'] = item[3]
        data['realtor name'] = item[4]
        data['property link'] = item[5]
        
        table_data.append(data)
    #logger.info(table_data)

    markdown = ""
    for item in table_data:
        markdown += "#### Property\n"
        for key, value in item.items():
            markdown += f"- **{key}**: {value}\n"
        markdown += "\n"

    with open(
        cfg.save_html_path / "savills_markdown.md", mode="wt", encoding="utf-8"
    ) as f:
        f.write(markdown)
    
    return markdown



def create_zip(url_response):
    soup = create_soup(url_response)
    property_names = find_name_property(soup)
    property_details = find_property_details(soup)
    realtor_names = find_realtor(soup)
    property_address = find_address(soup)
    property_links = find_property_link(soup)
    property_sizes = property_size(soup)
    list_properties = (list(zip_details(property_names, property_details, property_sizes, property_address, realtor_names, property_links)))
    markdown = create_markdown(list_properties)
    return markdown

if __name__ == "__main__":
    url_property = requests.get(
        f"https://search.savills.com/in/en/list?SearchList=Id_1243+Category_RegionCountyCountry&Tenure=GRS_T_B&SortOrder=SO_PCDD&Currency=INR&&Bedrooms=GRS_B_1&Bathrooms=-1&CarSpaces=-1&Receptions=-1&ResidentialSizeUnit=SquareFeet&LandAreaUnit=Acre&Category=GRS_CAT_RES&Shapes=W10"
    )
    # logger.info(find_realtor(url_property))
    # html_loader("C:/tmp/html_savills/savills.txt")
    zip_file =  create_zip(url_property)
    #soup = create_soup(url_property)
    #addr = find_address(soup)
    logger.info(zip_file)
    #create_markdown(list_properties)
