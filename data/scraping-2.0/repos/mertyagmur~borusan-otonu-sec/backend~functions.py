import requests
from bs4 import BeautifulSoup
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain

def scrape_automobile_info(url):
    # Send a GET request to the target URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the route using the provided CSS selector
        route_element = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div.container.p-0 > div:nth-child(2) > div > nav > ol > li.active > a")
        
        # Extract the href attribute value from the route element
        route_href = route_element.get('href')

        # Split the route by forward slashes
        route_parts = route_href.split('/')[1:]

        # Get car price
        car_price_element = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div.bg-gray-cont > div > div.d-sm-flex.pt-3 > div.col-sm-7.col-12.d-sm-flex.detail-summary > div.col-sm-6.col-12.text-center.text-sm-right > h5 > b")

        # Remove TL sign
        car_price = car_price_element.text.replace("\u20ba", "")

        # Get car photo
        car_photo_element = soup.find('img', {'data-index': str(0)})
        car_photo_src = car_photo_element.get("src")
        
        # Get technical specs
        technical_specs = {}
        specs_container = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div:nth-child(5) > div > div.col.col-12.detail-tab-content.technical-spesifications > div")
        if specs_container:
            # Find all nested div elements
            nested_divs = specs_container.find_all('div', class_='spesification')

            # Loop over the nested divs
            for i, div in enumerate(nested_divs):
                # Process each div as needed
                spec_name = div.select_one("div > div.spesification-inner-text > div:nth-child(1)")
                spec_content = div.select_one("div > div.spesification-inner-text > div:nth-child(2)")
                if not (spec_content.get("data-toggle")):
                    technical_specs[f"spec_{i}"] = {"name": spec_name.text.strip(), "content": spec_content.text.strip()}
                else:
                    technical_specs[f"spec_{i}"] = {"name": spec_name.text.strip(), "content": spec_content.get("data-content").strip()}
        else:
            print("Error: Unable to find the specs container.")

        # Get hardware packages
        hardware_packages = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div:nth-child(5) > div > div.col.col-12.detail-tab-content.hardware > div")
        feature_elements = hardware_packages.find_all("div", class_="hardware-inner-item")
        features = []
        for feature_el in feature_elements:
            features.append(feature_el.text.replace('"', "inç").strip())

        # Get expertise info
        damage_amount = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div:nth-child(5) > div > div.col.col-12.detail-tab-content.expertise > div.col.col-12.detail-tab-content.expertise > div > div:nth-child(1) > div > div.spesification-inner-text.d-flex.flex-row.flex-sm-column.justify-content-between.justify-content-sm-start > div:nth-child(2)")
        damage_amount = damage_amount.text.replace("\u20ba", "").strip()

        parts_container = soup.select_one("body > div.layout-content > div.detail.w-100 > div > div:nth-child(5) > div > div.col.col-12.detail-tab-content.expertise > div.row > div.col.col-12.col-sm-12.col-lg-5.expertise-left-container > div:nth-child(2)")
        parts = parts_container.find_all("div", class_="expertise-item")
        parts_dict = {}
        for part in parts:
            part_name = part.select_one("div > span").text
            part_conditions = part.find_all("div", class_="col-1")
            full_condition_str = ""
            for condition in part_conditions:
                condition_str = condition.select_one("span")
                if not condition_str:
                    full_condition_str += "Hasarsız"
                elif condition_str.text.strip() == "OG":
                    if full_condition_str != "":
                        full_condition_str += " ve Onarım Gerekli"
                    else:
                        full_condition_str += "Onarım Gerekli"
                elif condition_str.text.strip() == "D":
                    if full_condition_str != "":
                        full_condition_str += " ve Değişmiş"
                    else:
                        full_condition_str += "Değişmiş"
                elif condition_str.text.strip() == "B":
                    if full_condition_str != "":
                        full_condition_str += " ve Boyalı"
                    else:
                        full_condition_str += "Boyalı"
            parts_dict[part_name] = full_condition_str

        # Create a JSON object
        car_info = {
            "car_name": {
                "brand": route_parts[1],
                "model": route_parts[2],
                "package": route_parts[3]
            },
            "car_price": car_price,
            "car_photo": car_photo_src,
            "technical_specifications": technical_specs,
            "features": features,
            "expertise_info": {
                "damage_amount": damage_amount,
                "parts": parts_dict
            }
        }
        return car_info

    else:
        print(f"Error: Unable to fetch the page. Status code: {response.status_code}")

def clean_data(data1, data2):
    merged_json = {
        "car_1": data1,
        "car_2": data2
    }
    json_str = json.dumps(merged_json, ensure_ascii=False, indent=2)
    cleaned_json_str = json_str.replace('\n', '').replace('rn ', '').replace('\\', '').strip()
    cleaned_json_data = json.loads(cleaned_json_str)
    formatted_json_str = json.dumps(cleaned_json_data, indent=2, ensure_ascii=False)

    llm_output = langchain(formatted_json_str)
    return llm_output

def langchain(cars_json):
    load_dotenv(find_dotenv(), override=True)

    os.environ.get("OPENAI_API_KEY")    

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1900)
    
    template = '''
        You are a used car expert. 
        Compare two used cars for a non-expert audience by distilling complex automotive details 
        into a clear and accessible format. Focus on collecting specifications like engine size, 
        safety features, technology, and fuel efficiency. Organize the information into key categories 
        such as performance, safety, technology, and space. Clearly present the pros and cons in layman's 
        terms using relatable examples for real-world impact. Consider visual aids for better understanding
        and provide a comprehensive summary with cost considerations. Empower the audience to make 
        informed decisions based on individual needs and priorities, avoiding redundant feature listings. 
        Your text should be in Turkish. The text must be divided into the following sections:
        Introduction
        Performance
        Safety
        Technology
        Space
        Expertiz Bilgisi
        Cost Considerations
        Final Summary
        Provide creative and unique titles for each section don't just use the section names. 
        Include the names and models of the cars that you're comparing in the titles when appropriate.
        Try to use emojis in the titles.
        Wrap the titles and text in approppriate HTML tags. Don't write entire HTML, use <p> and <h> tags.
        Titles should always be wrapped inside h2 tags.
        Use third person plural voice
        Provide a comparison between {car_1} in {car_2}.
    '''
    cars_dict = json.loads(cars_json)
    car_1 = cars_dict["car_1"]
    car_2 = cars_dict["car_2"]

    prompt = PromptTemplate(
        input_variables=["car_1", "car_2"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"car_1": car_1, "car_2": car_2})
    print(output)
    return output