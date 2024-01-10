from pathlib import Path
import os
import etl.shared
import functools
import projsecrets

from langchain.document_loaders import BSHTMLLoader


def main(json_path, collection=None, db=None):
    """Calls the ETL pipeline using a JSON file with markdown file metadata.
    """
    import json

    with open(json_path) as f:
        markdown_corpus = json.load(f)

    for contentName, content in markdown_corpus.items():
        assert isinstance(content, list)

        """
        allurls = []
        alltitles = []
        for entry in content:
            allurls.append(entry["url"])
            alltitles.append(entry["title"])
        """

        documents = (
                etl.shared.unchunk(  # each lecture creates multiple documents, so we flatten
                map(functools.partial(to_documents), content))
            )

        chunked_documents = etl.shared.chunk_into(documents, 10)
        res = list(map(etl.shared.add_to_document_db, chunked_documents))
        res = res

#@stub.function(image=image)
def to_documents(content):
    title = content["title"]
    website_url = Path("data") / Path("markdowncontent") / content["url"]

    """
    text = get_text_from(website_url)
    headings, heading_slugs = get_target_headings_and_slugs(text)

    subtexts = split_by_headings(text, headings)
    headings, heading_slugs = [""] + headings, [""] + heading_slugs

    sources = [f"{website_url}#{heading}" for heading in heading_slugs]
    metadatas = [
        {
            "source": website_url,
            "heading": heading,
            "title": title,
            "full-title": f"{title} - {heading}",
        }
        for heading, website_url in zip(headings, sources)
    ]

    documents = [
        {"text": subtext, "metadata": metadata}
        for subtext, metadata in zip(subtexts, metadatas)
    ]
    """

    loader = BSHTMLLoader(website_url)
    data = loader.load()


    metadatas = [{
            "source" : os.path.basename(docData.metadata["source"]),
            "title" : docData.metadata["title"]}
        for docData in data]

    documents =  [
        {"text": docData.page_content.replace("\n", " ").encode("utf-8", errors="replace").decode(),
         "metadata": metadata}
        for docData, metadata in zip(data, metadatas)]

    documents = etl.shared.enrich_metadata(documents)

    return documents


#@stub.function(image=image)
def get_text_from(url):
    from smart_open import open

    with open(url) as f:
        contents = f.read()

    return contents


#@stub.function(image=image)
def get_target_headings_and_slugs(text):
    """Pull out headings from a markdown document and slugify them."""
    import mistune
    from slugify import slugify

    markdown_parser = mistune.create_markdown(renderer="ast")
    parsed_text = markdown_parser(text)

    heading_objects = [obj for obj in parsed_text if obj["type"] == "heading"]
    h2_objects = [obj for obj in heading_objects if obj["attrs"]["level"] == 2]

    targets = [
        obj
        for obj in h2_objects
        if not (obj["children"][0]["raw"].startswith("description: "))
    ]
    target_headings = [tgt["children"][0]["raw"] for tgt in targets]

    heading_slugs = [slugify(target_heading) for target_heading in target_headings]

    return target_headings, heading_slugs


def split_by_headings(text, headings):
    """Separate Markdown text by level-1 headings."""
    texts = []
    for heading in reversed(headings):
        text, section = text.split("# " + heading)
        texts.append(f"## {heading}{section}")
    texts.append(text)
    texts = list(reversed(texts))
    return texts

if __name__ == "__main__":
    main("data/webcontent.json")