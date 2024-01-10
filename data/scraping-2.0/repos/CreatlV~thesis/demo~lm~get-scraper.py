import openai
import os
from dotenv import load_dotenv
from dict_match import get_html_xpath_from_dict

from html_utils import clean_html, download_html_and_text
from openai_parser import extract_dictionary_from_text
from openai_parser import ModelType
from xpath_scraper import extract_data_from_html

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# url = "https://www.bygghemma.se/kok-och-bad/badrum/badrumsmobler/tvattstallsskap-och-kommod/tvattstallsskap-bathlife-lattsam-natur/p-1215685"
# url = "https://www.amazon.se/Intex-66644-Elektrisk-Pump-Gr%C3%A5/dp/B07FBC7QDL/ref=sr_1_1?pd_rd_r=00e37d0f-667f-472f-8106-63ff686032c9&pd_rd_w=ztvnK&pd_rd_wg=mDMPm&pf_rd_p=942a7617-af6c-47cf-a8fa-51b06ac5859b&pf_rd_r=GH0RCFQRYFZCN6XZEN3Y&qid=1686744353&refinements=p_36%3A-50000%2Cp_72%3A20692905031&s=sporting-goods&sr=1-1"
# url = "https://www.rum21.se/ikat-bangalore-utomhuskudde-50x50-cm?p=336447"
# url = "https://www.trendrum.se/oxford-matbord-220-cm-shabby-chic-mobelpolish"
url = "https://www.chilli.se/m%C3%B6bler/bord/matbord-k%C3%B6ksbord/matbord-navjot-120-cm-brun-p1766953-v1577363"

visible_text, html_string = download_html_and_text(url, use_cache=False)
cleaned_html = clean_html(html_string)

## Save cleaned_html to a file for debugging
with open("cleaned.html", "w", encoding="utf-8") as file:
    print(len(cleaned_html), len(html_string))
    file.write(cleaned_html)


model: ModelType = "gpt-4"
openai_response = extract_dictionary_from_text(model, visible_text)
print(openai_response)

## Load sample response from a file for debugging
# openai_response = json.loads(open("dict.json", "r", encoding="utf-8").read())
# print(openai_response)


xpath_dict = get_html_xpath_from_dict(openai_response, cleaned_html, keys_to_ignore=["currency"])
print(xpath_dict)

## Test scrape based on the xpath_dict
print()
print(extract_data_from_html(xpath_dict, html_string=cleaned_html))


