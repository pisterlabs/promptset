import csv
import re
import requests
from bs4 import BeautifulSoup
import openai
import utils
from hubspot import HubSpot
import tiktoken
from hubspot.crm.companies import PublicObjectSearchRequest, SimplePublicObjectInput


def extract_domain(email):
    domain = re.search("@[\w.]+", email)
    return domain.group()[1:] if domain else None

def scrape_front_page(domain):
    try:
        url = f"http://{domain}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except:
        return None

def categorize_domain(content):
    engine = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(model=engine, max_tokens=10, n=1,
                                            messages=[{"role": "system",
                                                       "content": "You are LLM that classifies the following website content as hosting, other or free mail service. "
                                                                  "Shared hosting, VPS hosting, WordPress Hosting, Dedicated Hosting, Cloud Hosting should be classified as hosting."
                                                                  "Free email services like Gmail, Yahoo, Outlook, Hotmail, AOL, etc. should be classified as free mail service. "
                                                                  "Parked domains, domain auctions and under construction pages should be marked as other. "
                                                                  "Everything else should be classified as other. "
                                                                  "Reply only with hosting, other or free"},
                                                      {"role":"user", "content": "classify the following website content: "+content}])
    category = response.choices[0].message["content"]

    return category




FREE_EMAILS = ()


def classify(email):
    return classify_domain(extract_domain(email))

def classify_domain(domain):
    ### convert domain to lower case
    if domain:
        domain = domain.lower()
        if domain in FREE_EMAILS:
            return "free email service, old"
        else:
            front_page = scrape_front_page(domain)
            if front_page is None:
                return "Unable to retrieve content"
            text = front_page.get_text()
            size = 5000
            while size > 4000:
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
                size = len(encoding.encode(text))
                if size > 4000:
                    text = text[:int(len(text)*1/3)]
            category = categorize_domain(text)
            return category
    else:
        return "No domain found in email address"


def search_contacts(api_key, list_id, limit=100):
    url = f'https://api.hubapi.com/contacts/v1/lists/{list_id}/contacts/all'
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    params = {
        'count': limit,
        'property': ['email', 'firstname', 'lastname', 'company', 'Email Domain'],  # Add or remove properties as needed
        'formSubmissionMode': 'none',
        'showListMemberships': 'false'
    }

    contacts = []

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if response.status_code != 200:
        raise Exception(f"Error: {data['message']} (Status code: {response.status_code})")

    contacts.extend(data['contacts'])
    return contacts


def search_companies(contact_ids, limit):
    companies = []
    public_object_search_request = PublicObjectSearchRequest(
        filter_groups=[
            {
                "filters": [
                    {
                        "propertyName": "associations.contact",
                        "operator": "IN",
                        "values": contact_ids
                    }
                ]
            }], limit=limit)

    response = client.crm.companies.search_api.do_search(public_object_search_request=public_object_search_request)
    companies.extend(response.results)
    return companies

def set_company_type(company_id, hosting):
    properties = {
        "hosting_type_tested": "true",
    }
    if hosting:
        properties["company_type"] = "Hosting"
    data = SimplePublicObjectInput(
        properties=properties
    )
    client.crm.companies.basic_api.update(company_id, simple_public_object_input=data)

def save_free_emails():
    with open('free_emails.txt', 'w') as f:
        for item in FREE_EMAILS:
            f.write(f'{item}\n')

def load_free_emails():
    with open('free_emails.txt', 'r') as f:
        return f.read().splitlines()


def save_comp_cat_as_csv(comp_cat):
    with open('comp_cat.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(comp_cat)

if __name__ == "__main__":
    openai.api_key = utils.load_openai_creds()['openai_api_key']
    hubspot_api_key = utils.load_hubspot_creds()
    client = HubSpot(access_token=hubspot_api_key)

    FREE_EMAILS = load_free_emails()
    list_id = '3621'
    limit = 100
    contacts = search_contacts(hubspot_api_key, list_id, limit=limit)
    contact_ids = []
    for contact in contacts:
        contact_ids.append(str(contact['vid']))

    companies = search_companies(contact_ids, limit=limit)
    comp_cat = []
    for company in companies:
        domain = company.properties['domain']
        cat = classify_domain(domain)
        set_company_type(company.id, "hosting" in cat.lower())
        if "free mail" in cat.lower():
            if domain not in FREE_EMAILS:
                FREE_EMAILS.append(company.properties['domain'])
                save_free_emails()
        comp_cat.append([company.id, company.properties['domain'], cat])
        print(company.id, company.properties['domain'], cat)
    save_comp_cat_as_csv(comp_cat)