import markdownify
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownHeaderTextSplitter


class MyMarkdownConverter(markdownify.MarkdownConverter):

    def convert_a(self, el, text, convert_as_inline):
        
        text = el.text
        for child in el.children:
            text += super().process_tag(child,convert_as_inline)
        return text

def process_soup(soup : BeautifulSoup) -> str:

    for useless_tag in soup("script", "style", "meta", "head"):
        useless_tag.decompose()

    # return MyMarkdownConverter().convert(str(soup))
    return markdownify.markdownify(str(soup))

_default_headers_to_split_on = [
    ("#", "Title"),
    ("##", "Paragraph title"),
    ("###", "Subparagraph title")
] 

def split_markdown(markdown : str, headers_to_split_on = _default_headers_to_split_on):

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    return splitter.split_text(markdown)