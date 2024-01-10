
import pandas as pd
import openai
from IPython.display import display

# list all open ai models
engines = openai.Engine.list()
pd = pd.DataFrame(engines['data'])
display(pd[['id', 'owner']])

'''
                               id            owner
0                         babbage           openai
1                         davinci           openai
2           text-davinci-edit-001           openai
3        babbage-code-search-code       openai-dev
4     text-similarity-babbage-001       openai-dev
5                   gpt-3.5-turbo           openai
6           code-davinci-edit-001           openai
7                text-davinci-001           openai
8                             ada           openai
9        babbage-code-search-text       openai-dev
10             babbage-similarity       openai-dev
11             gpt-3.5-turbo-0301           openai
12   code-search-babbage-text-001       openai-dev
13                 text-curie-001           openai
14                      whisper-1  openai-internal
15   code-search-babbage-code-001       openai-dev
16               text-davinci-003  openai-internal
17                   text-ada-001           openai
18         text-embedding-ada-002  openai-internal
19        text-similarity-ada-001       openai-dev
20            curie-instruct-beta           openai
21           ada-code-search-code       openai-dev
22                 ada-similarity       openai-dev
23       code-search-ada-text-001       openai-dev
24      text-search-ada-query-001       openai-dev
25        davinci-search-document       openai-dev
26           ada-code-search-text       openai-dev
27        text-search-ada-doc-001       openai-dev
28          davinci-instruct-beta           openai
29      text-similarity-curie-001       openai-dev
30       code-search-ada-code-001       openai-dev
31               ada-search-query       openai-dev
32  text-search-davinci-query-001       openai-dev
33             curie-search-query       openai-dev
34           davinci-search-query       openai-dev
35        babbage-search-document       openai-dev
36            ada-search-document       openai-dev
37    text-search-curie-query-001       openai-dev
38    text-search-babbage-doc-001       openai-dev
39          curie-search-document       openai-dev
40      text-search-curie-doc-001       openai-dev
41           babbage-search-query       openai-dev
42               text-babbage-001           openai
43    text-search-davinci-doc-001       openai-dev
44  text-search-babbage-query-001       openai-dev
45               curie-similarity       openai-dev
46                          curie           openai
47    text-similarity-davinci-001       openai-dev
48               text-davinci-002           openai
49             davinci-similarity       openai-dev
'''


from openai.embeddings_utils import get_embedding

text = "让我们来算算Embedding"

embedding_ada = get_embedding(text, engine="text-embedding-ada-002")
print("embedding-ada: ", len(embedding_ada))

similarity_ada = get_embedding(text, engine="text-similarity-ada-001")
print("similarity-ada: ", len(similarity_ada))

babbage_similarity = get_embedding(text, engine="babbage-similarity")
print("babbage-similarity: ", len(babbage_similarity))

babbage_search_query = get_embedding(text, engine="text-search-babbage-query-001")
print("search-babbage-query: ", len(babbage_search_query))

curie = get_embedding(text, engine="curie-similarity")
print("curie-similarity: ", len(curie))

davinci = get_embedding(text, engine="text-similarity-davinci-001")
print("davinci-similarity: ", len(davinci))

# 可以看到，最小的 ada-similarity 只有 1024 维，而最大的 davinci-similarity 则有 12288 维，所以它们的效果和价格不同也是可以理解的了。