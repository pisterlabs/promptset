"""
Crawling from Wikipedia.
c.r. OpenAI Cookbook
"""
import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens


# ##########################################
# Temp arguments
# ##########################################
search_term = 'chemistry'
MAX_TOKENS = 1600
SAVE_PATH = "../collections/wiki/college_chemistry.csv"
WIKI_SITE = 'en.wikipedia.org'
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000

# ##########################################
# Helper Functions
# ##########################################
# ==========================================
# Search the relevant category
# ==========================================
def search_categories(site, search_term):
    """
    Search for categories using the search API.
    """

    return [page['title'] for page in site.search(search_term, namespace=14)]


def all_categories(site: mwclient.Site) -> list[str]:
    """
    Return a list of all category names on a given Wiki site.
    """

    categories = []
    for category in site.allcategories():
        categories.append(category.name)
        if len(categories) > 100:
            break
    return categories


# ==========================================
# Search the relevant category
# ==========================================
def titles_from_category(category: mwclient.listing.Category,
                         max_depth: int,
                         limit: int=3000) -> set[str]:
    """
    Return a set of page titles in a given Wiki category and its subcategories.
    """

    titles = set()

    for cm in category.members():
        if type(cm) == mwclient.page.Page:
            # ^type() used instead of isinstance() to catch match w/ no inheritance
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)

        if len(titles) > limit:
            return titles

    return titles


# ==========================================
# Chunk the documents
# ==========================================
# define functions to split Wikipedia pages into sections
SECTIONS_TO_IGNORE = ["See also", "References", "External links",
                      "Further reading", "Footnotes", "Bibliography",
                      "Sources", "Citations", "Literature", "Footnotes",
                      "Notes and references", "Photo gallery",
                      "Works cited", "Photos", "Gallery", "Notes",
                      "References and sources", "References and notes"]


def all_subsections_from_section(section: mwparserfromhell.wikicode.Wikicode,
                                 parent_titles: list[str],
                                 sections_to_ignore: set[str],
                                 ) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia section, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """

    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []

    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]

    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]

        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection,
                                                        titles,
                                                        sections_to_ignore))

        return results


def all_subsections_from_title(title: str,
                               sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
                               site_name: str = WIKI_SITE,
                               ) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """

    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]

    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)

    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection,
                                                    [title],
                                                    sections_to_ignore))
    return results


# ==========================================
# Clearn the texts and filtering
# ==========================================
# clean text
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)


# filter out short/blank sections
def keep_section(section: tuple[list[str], str]) -> bool:
    """Return True if the section should be kept, False otherwise."""
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True

# ==========================================
# Tokenization
# ==========================================
def num_tokens(text: str,
               model: str = GPT_MODEL) -> int:

    """
    Return the number of tokens in a string.
    """

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """
    Split a string in two, on a delimiter, trying to balance tokens on each side.
    """

    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found

    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point

    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)

            if diff >= best_diff:
                break
            else:
                best_diff = diff

        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(string: str,
                     model: str,
                     max_tokens: int,
                     print_warning: bool = True) -> str:
    """
    Truncate a string to a maximum number of tokens.
    """

    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])

    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")

    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """

    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)

    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]

    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]

    # otherwise, split in half and recurse
    else:
        titles, text = subsection

        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)

            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue

            else:
                # recurse on each half
                results = []

                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


# ##########################################
# Main function
# ##########################################
# ==========================================
# 0. Search the relevant category
# ==========================================
site = mwclient.Site(WIKI_SITE)
found_categories = search_categories(site, search_term)
CATEGORY_TITLE = found_categories[0]

# ==========================================
# 1. Search the relevant articles from the cate
# ==========================================
site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
titles = titles_from_category(category_page, max_depth=1)
# ^note: max_depth=1 means we go one level deep in the category tree
print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")

# ==========================================
# 2. Chunk the documents
# ==========================================
# split pages into sections
# may take ~1 minute per 100 articles
wikipedia_sections = []
for title in titles:
    wikipedia_sections.extend(all_subsections_from_title(title))
print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")

# ==========================================
# 3. Clearn the texts and filtering
# ==========================================
# clean text
wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]
original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections")
print(f"leaving {len(wikipedia_sections)} sections.")

# ==========================================
# 4. Chunk and Save
# ==========================================
# split sections into chunks
wikipedia_strings = []
for section in wikipedia_sections:
    wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")

# Calculate Embeddings
embeddings = []
for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = wikipedia_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})
df.to_csv(SAVE_PATH, index=False)
