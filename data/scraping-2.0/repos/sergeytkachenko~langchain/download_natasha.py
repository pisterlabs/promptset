from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import AsyncHtmlLoader

from langchain.text_splitter import MarkdownTextSplitter

from html.parser import HTMLParser
import numpy as np
import json
from bs4 import BeautifulSoup
import re
import markdownify
from markdownify import markdownify as md

from langchain.document_transformers import Html2TextTransformer

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
visited = []
stoplinks = [
    "https://academy.creatio.com/docs/",
    'https://academy.creatio.com/docs/?vid_1=1',
]
links = ["https://academy.creatio.com/docs/8.x/setup-and-administration/on_site_deployment/containerized_components/global_search",
"https://academy.creatio.com/docs/8.x/setup-and-administration/on_site_deployment/containerized_components/global_search_and_deduplication_faq",
"https://academy.creatio.com/docs/8.x/setup-and-administration/on_site_deployment/containerized_components/bulk_duplicate_search",
"https://academy.creatio.com/docs/8.x/setup-and-administration/on_site_deployment/containerized_components/email_listener_synchronization_service",
"https://academy.creatio.com/docs/8.x/dev/development-on-creatio-platform/platform_customization/classic_ui/telephony_integration/overview",
"https://academy.creatio.com/docs/8.x/dev/development-on-creatio-platform/platform_customization/classic_ui/telephony_integration/webitel",
"https://academy.creatio.com/docs/8.x/dev/development-on-creatio-platform/platform_customization/classic_ui/telephony_integration/asterisk",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/add_imap_smtp_email_provider",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/google_mail_contacts_and_calendar/register_creatio_application_in_gsuite",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/google_mail_contacts_and_calendar/synchronize_contacts_and_activities_with_google",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/google_mail_contacts_and_calendar/delete_your_google_account_from_creatio",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/microsoft_email_contacts_and_calendar/set_up_the_ms_exchange_and_microsoft_365_services",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/microsoft_email_contacts_and_calendar/synchronizing_creatio_calendar_with_ms_exchange_and_microsoft_365_calendars",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/microsoft_email_contacts_and_calendar/set_up_oauth_authentication_for_ms_office_365",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/mailbox_setup/set_up_a_secure_mailbox_connection",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/mailbox_setup/set_up_a_personal_mailbox",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/mailbox_setup/configure_a_shared_mailbox",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/mailbox_setup/email_account_individual_settings",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/mailbox_setup/mailbox_setup_faq",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_webitel",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_asterisk",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_avaya",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_cisco_finesse",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_tapi",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/set_up_integration_with_callway",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/configure_a_wss_phone_service_connection",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/creatio_phone_integration_faq",
"https://academy.creatio.com/docs/8.x/no-code-customization/base_integrations/phone_integration_connectors/feature_comparison_for_supported_phone_systems%20copy"]

final_links = []
files_map = {}

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for x in attrs:
                if x[0] == "href" and x[1].startswith("/docs/"):
                    links.append("https://academy.creatio.com" + x[1])
                    break


parser = MyHTMLParser()

def parse(links):
    links_unique = list(np.unique(links))
    for x in links_unique:
        visited.append(x)
    links1 = []

    all_links = []
    for link in links_unique:
        print(link)
        loader = AsyncHtmlLoader(link)
        docs = loader.load()

        soup = BeautifulSoup(docs[0].page_content, 'lxml')
        documentationbody = soup.find("div", {"class": "theme-doc-markdown"})
        if documentationbody is not None:
            h1 = soup.find("div", {"class": "theme-doc-markdown"}).find('h1').text
            title = re.sub(' ', '-', h1)
            title = re.sub('[^a-zA-Z0-9-_]', '', title)
            h1 = re.sub('[/.%$&*()#!@]', '-', title)
            files_map[link] = title
            f = open('./md/' + h1 + ".txt", "w", encoding="utf-8")
            markdown = md(str(documentationbody))
            # markdown_text = markdownify.markdownify(documentationbody.prettify(), encodings='utf-8', strip=['script'])
            f.write(markdown)
            f.close()
    return all_links


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # loader = AsyncHtmlLoader([
    #     "https://academy.creatio.com/docs/user/on_site_deployment/containerized_components/global_search_shortcut/global_search",
    # ])
    # # loader = UnstructuredMarkdownLoader("example_data/fake-content.html")
    # docs = loader.load()
    # # html2text = Html2TextTransformer()
    # # docs_transformed = html2text.transform_documents(docs)
    # # print(docs_transformed)
    # parser.feed(docs[0].page_content)
    # # markdown_text = markdownify.markdownify(docs[0].page_content)
    # # print(docs)
    #
    # # loader1 = TextLoader("gs.md", encoding="utf-8")
    # # docs1 = loader1.load()
    # # print(docs1)
    # links2 = list(filter(lambda x: x.startswith("https://academy.creatio.com/docs/user/login") == False, links))
    parse(links)
    # links3 = parse(links2)
    # links4 = parse(links3)
    # links5 = parse(links4)
    # print(links5)

    # text_splitter = MarkdownTextSplitter()
    # docssplit = text_splitter.split_documents(docs1)
    # print(docssplit)
    #
    # f = open("final_links.json", "w", encoding="utf-8")
    # lll = list(np.unique(list(set(final_links))))
    # json.dump(lll, f)
    #
    # f = open("files_map.json", "w", encoding="utf-8")
    # json.dump(files_map, f)
    # f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
