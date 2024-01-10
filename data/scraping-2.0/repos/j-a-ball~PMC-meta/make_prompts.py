__author__ = "Jon Ball"
__version__ = "December 2023"

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from vectordb_setup import hf_embed, openai_embed
from tqdm import tqdm
import chromadb
import jinja2
import torch
import random
import os


def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # Access local chroma client
    persistent_client = chromadb.PersistentClient(path="chroma")
    # Load articles
    articles = load_articles()
    # Load embeddings
    print("Loading HF embeddings...")
    hf_ef = hf_embed("Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit")
    openai_ef = openai_embed("text-embedding-ada-002")
    print("   ...HF embeddings loaded.")
    # Use hf retrieval models to generate prompts
    print("Generating prompts with HuggingFace SGPT retrieval...")
    collection_name = "2_7B"
    output_dir = f"prompts/{collection_name}"
    vectordb = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=hf_ef
        )
    article_loop(articles, vectordb, output_dir)
    print("   ...prompts generated.")
    # Use openai retrieval model to generate prompts
    print("Generating prompts with OpenAI retrieval...")
    collection_name = "openai"
    output_dir = f"prompts/{collection_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vectordb = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=openai_ef
        )
    article_loop(articles, vectordb, output_dir)
    print("   ...prompts generated.")
    print("Done.")


def article_loop(articles, vectordb, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename, article in tqdm(articles):
        checklist = vectordb.similarity_search(article, 1)[0].metadata["checklist"]
        # generate prompt with jinja2
        jinjitsu = jinjaLoader("prompts", "checklist.prompt")
        templateVars = {"checklist": checklist, "article": article}
        prompt = jinjitsu.render(templateVars)
        # write prompt to file
        with open(f"{output_dir}/{filename[:-4]}.prompt", "w") as outfile:
            outfile.write(prompt)


def load_articles():
    print("Loading articles...")
    articles = []
    for dirname, dirpath, filenames in os.walk("articles"):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            with open(os.path.join(dirname, filename), "r") as infile:
                article = infile.read()
            articles.append((filename, article))
    print(f"   ...loaded {len(articles)} articles.")
    return articles


class jinjaLoader():
    # jinja2 template renderer
    def __init__(self, template_dir, template_file):
        self.templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.templateEnv = jinja2.Environment( loader=self.templateLoader )
        self.template = self.templateEnv.get_template( template_file )

    def render(self, templateVars):
        return self.template.render( templateVars )
    

if __name__ == "__main__":
    main()