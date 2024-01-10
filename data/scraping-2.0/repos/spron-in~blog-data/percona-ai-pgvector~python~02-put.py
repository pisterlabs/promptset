# create vector embedding from string
from sentence_transformers import SentenceTransformer
import config
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import re
from bs4 import BeautifulSoup
import xmltodict
import requests
import psycopg2
from pgvector.psycopg2 import register_vector


model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
def create_embedding(content):
    embeddings = model.encode([content], device='cuda', show_progress_bar=True)
    return(embeddings)

conn = psycopg2.connect(
    user=config.PGUSER,
    password=config.PGPASSWORD,
    database=config.PGDATABASE,
    host=config.PGHOST,
    port=config.PGPORT,
)
cur = conn.cursor()
cur.execute("SET search_path TO " + 'test')
register_vector(conn)

def put_embedding(url, text, pgcur):

	embeddings = create_embedding(text)
	cur.execute('INSERT INTO perconavec (content, url, embedding) VALUES (%s,%s,%s)', (text, url, embeddings[0],))

pages = []
max_pages = 20
#########################
# Parsing Percona Blogs #
#########################

# from blog post - remove noisy divs
def extract_text_from_blog(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")

    for div in soup.find_all("div", {"id": "jp-relatedposts"}):
        div.decompose()
    for div in soup.find_all("div", {"class": "share-wrap"}):
        div.decompose()
    for div in soup.find_all("div", {"class": "comments-sec"}):
        div.decompose()

    text = soup.find("div", {"class": "blog-content-inner"}).get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def get_blog_chunks(content):
	
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
	chunks = text_splitter.create_documents([content])
	return chunks

r = requests.get("https://www.percona.com/blog/sitemap_index.xml")
xml = r.text
rootxml = xmltodict.parse(xml)

for xmlurl in rootxml['sitemapindex']['sitemap']:
	r = requests.get(xmlurl['loc'])
	xml = r.text
	raw = xmltodict.parse(xml)
	for info in raw['urlset']['url']:
		if len(pages) == max_pages:
			for page in pages:
				chunks = get_blog_chunks(page['text'])
				for chunk in chunks:
					print(page['source'])
					put_embedding(page['source'], chunk.page_content, cur)
					conn.commit()
			pages = []
		url = info['loc']
		if 'https://www.percona.com/blog/' in url:
			pages.append({'text': extract_text_from_blog(url), 'source': url})

if pages:
	chunks = get_blog_chunks(page['text'])
	for chunk in chunks:
		print(page['source'])
		put_embedding(page['source'], chunk.page_content, cur)
		conn.commit()
	pages = []
########################
# Parsing Percona Docs #
########################

# extract md paths and urls
def get_md_docs(doc):

	md_docs = []
	api_url = 'https://api.github.com/repos/%s/git/trees/%s?recursive=1' % (doc['repo'], doc['branch'])
	r = requests.get(api_url)
	for file in r.json()['tree']:
		m = re.search(r'docs/.*\.md', file['path'])
		if m:
			md_docs.append({'path': file['path'], 'url': file['url']})
	return md_docs

def get_md_content(url):
	md = requests.get(url).text
	return md

def get_doc_chunks(content):
	markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=0)
	chunks = markdown_splitter.create_documents([content])
	return chunks

# parse docs
docs = [
	{'repo': 'percona-platform/portal-doc', 'branch': 'main'},
	{'repo': 'percona/pmm-doc', 'branch': 'main'},
	{'repo': 'percona/pdmysql-docs', 'branch': 'innovation-release'},
	{'repo': 'percona/pxc-docs', 'branch': '8.0'},
	{'repo': 'percona/pxb-docs', 'branch': 'innovation-release'},
	{'repo': 'percona/proxysql-admin-tool-doc', 'branch': 'main'},
	{'repo': 'percona/distmongo-docs', 'branch': '7.0'},
	{'repo': 'percona/psmdb-docs', 'branch': '7.0'},
	{'repo': 'percona/pbm-docs', 'branch': 'main'},
	{'repo': 'percona/postgresql-docs', 'branch': '16'},
	{'repo': 'percona/k8sps-docs', 'branch': 'main'},
	{'repo': 'percona/k8spsmdb-docs', 'branch': 'main'},
	{'repo': 'percona/k8spxc-docs', 'branch': 'main'},
	{'repo': 'percona/k8spg-docs', 'branch': 'main'},
	{'repo': 'percona/everest-doc', 'branch': 'main'},
	{'repo': 'percona/psmysql-docs', 'branch': 'innovation-release'}
]
pages = []
for doc in docs:

	md_docs = get_md_docs(doc)
	for md_doc in md_docs:
		url = "https://raw.githubusercontent.com/%s/%s/%s" % (doc['repo'], doc['branch'], md_doc['path'])
		pages.append({'source': "https://raw.githubusercontent.com/%s/%s/%s" % (doc['repo'], doc['branch'], md_doc['path']), 'text': get_md_content(url)})
		if len(pages) == max_pages:
			for page in pages:
				chunks = get_doc_chunks(page['text'])
				for chunk in chunks:
					print(page['source'])
					put_embedding(page['source'], chunk.page_content, cur)
					conn.commit()
			pages = []

conn.commit()
cur.close()
conn.close()
