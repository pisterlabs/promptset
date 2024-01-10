from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
load_dotenv()
import os

from pyzotero import zotero

library_type = 'user'
library_id = os.environ.get("ZOTERO_USER_ID")
zotero_api_key = os.environ.get("ZOTERO_API_KEY")

zot = zotero.Zotero(library_id, library_type, zotero_api_key)
collections = zot.collections()

col = zot.collection('IRF6FT7U')
#print('Collection: %s | Key: %s' % (col['data']['name'], col['data']['key']))
items = zot.collection_items('IRF6FT7U')
#for collection in collections:
#  print('Collection: %s | Key: %s' % (collection['data']['name'], collection['data']['key']))
attach_dir = './pdfs'
if not os.path.exists(attach_dir):
    os.makedirs(attach_dir)
    
for item in items:
    
    if item['data']['itemType'] == 'journalArticle':
        pass
    if 'contentType' in item['data'] and item['data']['contentType'] == 'application/pdf':
        
        print(item['data'])
        filepath =  attach_dir + '/' + item['data']['filename']

        if not os.path.exists(filepath):
          zot.dump(item['data']['key'], item['data']['filename'],'./pdfs')

        
  
     
  #print('Item Type: %s | Key: %s | Title: %s' % (item['data']['itemType'], item['data']['key'],item['data']['title']))

'''
items = zot.top(limit=5)
# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
  print('Item Type: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))
'''

openai_api_key=os.environ.get("OPENAI_API_KEY")
#print("api_key",openai_api_key)

client = OpenAI()

model_list = client.models.list()
for model in model_list:
    pass
    #print("id:", model.id)

exit()


completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:personal::8KLC3aEp",
  messages=[
    {"role": "system", "content": "You are an expert on trilobite taphonomy. Answer with citation in the sentence if possible."},
    {"role": "user", "content": "Please summarize Hesselbo's (1987) study on trilobite taphonomy."},
  ]
)

print(completion.choices[0].message) 
