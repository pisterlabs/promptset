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
from langchain.document_transformers import Html2TextTransformer

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
visited = []
stoplinks = [
    "https://academy.creatio.com/docs/",
    'https://academy.creatio.com/docs/?vid_1=1',
]
links = []
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

    class MyHTMLParser1(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for x in attrs:
                    if x[0] == "href" and x[1].startswith("/docs/"):
                        links1.append("https://academy.creatio.com" + x[1])
                        break

    all_links = []
    for link in links_unique:
        print(link)
        loader = AsyncHtmlLoader(link)
        docs = loader.load()

        soup = BeautifulSoup(docs[0].page_content, 'lxml')
        documentationbody = soup.find("div", {"class": "block-field-blocknodeuser-documentationbody"})
        if documentationbody is not None:
            title = soup.find('head').find('title').text
            title = re.sub(' ', '-', title)
            title = re.sub('[^a-zA-Z0-9-_]', '', title)
            files_map[link] = title
            f = open('./docs/' + title + ".md", "w", encoding="utf-8")
            markdown_text = markdownify.markdownify(documentationbody.prettify(), encodings='utf-8', strip=['script'])
            f.write(markdown_text)
            f.close()

        parser1 = MyHTMLParser1()
        parser1.feed(docs[0].page_content)
        list1 = list(filter(lambda x: x.startswith("https://academy.creatio.com/docs/user/login") == False, links1))
        list1 = list(filter(lambda x: x.startswith("https://academy.creatio.com/docs/doc/print/pdf") == False, list1))
        ll = np.unique(list(set(list1) - set(links_unique) - set(stoplinks) - set(visited)))
        for l in ll:
            all_links.append(l)
            final_links.append(l)
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
    new_links = None
    for x in range(1000):
        start = [
            "https://academy.creatio.com/docs/user/platform_basics/user_interface/classic_ui_overview/creatio_interface"]
        if new_links is None:
            new_links = parse(start)
        else:
            new_links = parse(new_links)
    print(final_links)
    # links3 = parse(links2)
    # links4 = parse(links3)
    # links5 = parse(links4)
    # print(links5)

    # text_splitter = MarkdownTextSplitter()
    # docssplit = text_splitter.split_documents(docs1)
    # print(docssplit)

    f = open("final_links.json", "w", encoding="utf-8")
    lll = list(np.unique(list(set(final_links))))
    json.dump(lll, f)

    f = open("files_map.json", "w", encoding="utf-8")
    json.dump(files_map, f)
    f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
