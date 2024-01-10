import os
import openai
from config import key

openai.api_key = key
response = openai.Image.create(prompt="a white siamese cat",
                               n=1,
                               size="1024x1024")
image_url = response['data'][0]['url']
print(image_url)



{created: 1686647266, data: Array(3)}
created
: 
1686647266
data
: 
Array(3)
0
: 
{url: 'https://oaidalleapiprodscus.blob.core.windows.net/…=hdCEF%2BHpLRDBMFSGj2Acb1KvKEqIf4OGwmayaaaMaqQ%3D'}
1
: 
{url: 'https://oaidalleapiprodscus.blob.core.windows.net/…ig=eQ1EHQ4/iRYVTQMJO/zXipdO6bzA5ivbgPRciiKiCSU%3D'}
2
: 
{url: 'https://oaidalleapiprodscus.blob.core.windows.net/…ig=o2gvtu0kzZQJYDnTpnBuEpUQah/GKKwHtqbQxdOjlmo%3D'}
length
: 
3
[[Prototype]]
: 
Array(0)
[[Prototype]]
: 
Object