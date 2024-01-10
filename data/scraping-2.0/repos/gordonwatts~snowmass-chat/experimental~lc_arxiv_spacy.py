# Try to download and "cache" the data loaded from a document on the archive

# from langchain.document_loaders import ArxivLoader
from pathlib import Path
import pickle
import spacy

from chathelper.lc_experimental.archive_loader import ArxivLoader


def sanitize(n: str) -> str:
    return n.replace(":", "_").replace(".", "_")


document_name = "id:2109.10905"

downloaded = Path(f"./{sanitize(document_name)}.pkl")
if not downloaded.exists():
    loader = ArxivLoader(
        document_name,
        load_all_available_meta=True,
        doc_content_chars_max=None,
        keep_pdf=True,
    )
    data = loader.load()

    print(len(data))
    assert len(data) == 1

    good_doc = data[0]

    with downloaded.open("wb") as f:
        pickle.dump(good_doc, f)
else:
    with downloaded.open("rb") as f:
        good_doc = pickle.load(f)

# And now read it back in to test that pickle round trip worked.
print(f"content len: {len(good_doc.page_content)}")

# Now lets see how this noun thing works.
models = {"lg": "en_core_web_lg", "md": "en_core_web_md", "sm": "en_core_web_sm"}
nlp = {name: spacy.load(m) for name, m in models.items()}

docs = {name: nlp(good_doc.page_content) for name, nlp in nlp.items()}

good_nouns = {
    name: {
        token.text
        for token in doc
        if len(token.text) > 2
        and token.pos_ == "PROPN"
        and token.is_oov
        and not token.text.lower().startswith("arxiv:")
    }
    for name, doc in docs.items()
}

for name, nouns in good_nouns.items():
    print(f"{name}: {len(nouns)}")
    print(list(nouns)[:10])

# Lets look at the difference in names found by the medium model and the large model.
print("lg - md")
print(good_nouns["lg"] - good_nouns["md"])
print("md - lg")
print(good_nouns["md"] - good_nouns["lg"])

# Finally, lets look at entities in the large model:
print("Organizations (lg):")
organizations = {ent.text for ent in docs["lg"].ents if ent.label_ == "ORG"}
for o in sorted([o.replace("\n", " ") for o in organizations]):
    if not o.lower().startswith("arxiv:"):
        print(f"  {o}")

# This did not turn up MATHUSLA, but it did turn up FASER and all its
# variations well! This might be a good way to do this, actually.

# for ent_tokens in docs["lg"].ents:
#     if ent_tokens.label_ == "ORG":
#         print(f"  {ent_tokens.text}: {ent_tokens.label_}")
