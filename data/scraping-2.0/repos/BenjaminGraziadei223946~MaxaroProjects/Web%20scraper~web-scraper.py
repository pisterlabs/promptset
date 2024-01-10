import requests
from bs4 import BeautifulSoup
import openai
import streamlit as st
import json
import time


st.title("Product Text Generator")

main_page = "https://www.maxaro.nl"
openai.api_type = "azure"
openai.api_key = "54a267e072934050a8df635e4f6da7b5"
openai.api_base = "https://maxbotai.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
counter = st.empty()
type = st.empty()
list_of_links = []

def generate_description(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Parsing the content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')

        product_title = soup.find('h1', class_='product-header__title').get_text().strip()

        specifications_container = soup.find('div', id="specifications")
        specs = specifications_container.find('div', class_="product-detail-specifications").get_text().strip()


        benefits_container = soup.find('div', id="benefits")
        benefits = benefits_container.find('div', class_="product-detail-section__content").get_text().strip() if benefits_container else "Benefits not found"
        
        text = f"{product_title}: {specs}. {benefits}"

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "Maak een overtuigende en positieve productbeschrijving voor het volgende artikel. Benadruk de belangrijkste kenmerken, voordelen en onderscheidende kenmerken. Gebruik duidelijke en begrijpelijke taal om de lezer te boeien. Stel je voor dat je tegen een potentiële klant spreekt die op zoek is naar de beste kwaliteiten van het product. Maak de beschrijving ongeveer 150-200 woorden."},
                {"role": "user", "content": f"{text}"}
            ]
        )
        st.write(response['choices'][0]['message']['content'])
    else:
        print(f"Error accessing the webpage: Status code {response.status_code}")

def is_button_disabled(button):
    if button.get('disabled'):
        return True
    if 'is-disabled' in button.get('class', ''):
        return True
    return False

def get_category_links(url, max_attempts=3, delay=3):                                                               # Get the links of the categories on Mainpage
    for attempt in range(max_attempts):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            category_container = soup.find('div', class_='categories-with-icon__container')
            if category_container:
                category_link_elements = category_container.find_all('a', class_='categories-with-icon__item')
                if category_link_elements:
                    return [link['href'] for link in category_link_elements if 'href' in link.attrs]                # return the links of the categories
        time.sleep(delay)
    return None

def get_sub_links(url, max_attempts=3, delay=3):                                                                   # Get the links of the subcategories inside the categories           
    for attempt in range(max_attempts):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            sub_container = soup.find('div', class_='categories-with-image')
            if sub_container:
                sub_link_elements = sub_container.find_all('a', class_='categories-with-image__item')             
                if sub_link_elements:
                    return [link['href'] for link in sub_link_elements if 'href' in link.attrs]                    # return the links of the subcategories
            else:
                return None
        time.sleep(delay)
    return None

def get_product_links(url, max_attempts=3, delay=3):                                                                 # Get the links of the products on given page
    for attempt in range(max_attempts):
        while True:                                                                                                 # Loop as long as there are more pages
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                product_container = soup.find_all('div', class_='column is-6-mobile is-4-tablet')                   # Find all the products on the page
                if product_container:
                    list_of_links.extend([product.find('a')['href'] for product in product_container if product.find('a')]) # Add the links to the list
                        
                    next_page_span = soup.find('span', class_='pagination__button-text', text='Volgende')           # Find the next page button text
                    next_page_button =  next_page_span.find_parent('a') if next_page_span else None                 # Find the next page button parent
                    if next_page_button and not is_button_disabled(next_page_button):                               # Check if the button is disabled and if not, get the link
                        url = main_page + next_page_button['href']                                                  # Update the url          
                        type.write(url)
                    else:                                                                                           # If the button is disabled, break the loop                        
                        break

            time.sleep(delay)
    return None

def get_links(main_page):

    categories = get_category_links(main_page)

    for category in categories:
        category_url = main_page + category
        subcategories = get_sub_links(category_url)

        if subcategories:
            for sub_link in subcategories:
                sub_url = main_page + sub_link
                get_product_links(sub_url)

        else:
            get_product_links(category_url)

    return None

get_links(main_page)
url = st.selectbox("Select a product", list_of_links)
if st.button("Generate"):
    #generate_description(url)
    st.write(list_of_links)
    with open('links.json', 'w') as file:
        json.dump(list_of_links, file)

    
