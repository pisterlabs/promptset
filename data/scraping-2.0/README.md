# Scraping using Playwright

usage: `python3 scraper.py`

WARNING: The library, and the GitHub username and password are hardcoded in the script. Please change them before running the script.

### Target libraries
libraries = ["openai", "llamaindex", "cohere", "guidance", "anthropic", "langchain"]

### Results from 11/30/2023

Result Counts from Github (Collected Manually)
    guidance: 1.9k
    anthropic: 1.8k
    llamaindex: 117
    cohere: 5.8k
    openai: 97.8k
    langchain: 64.5k

Library: cohere
        Total number of results: 5414
        Total number of hrefs: 4635
Library: guidance
        Total number of results: 1582
        Total number of hrefs: 1259
Library: anthropic
        Total number of results: 1657
        Total number of hrefs: 1297
Library: llamaindex
        Total number of results: 116
        Total number of hrefs: 122
Library: langchain
        Total number of results: 55114
        Total number of hrefs: 42374
Library: openai
        Total number of results: 77702
        Total number of hrefs: 55214

Total number of files DOWNLOADED: 93134

- Update:
I pushed all the newly collected data on Tuesday night. I forgot to provide an update since then. My bad. 😅 

```
Result Counts from Github (Collected Manually)
    guidance: 1.9k
    anthropic: 1.8k
    llamaindex: 117
    cohere: 5.8k
    openai: 97.8k
    langchain: 64.5k

Library: cohere
        Total number of results: 5414
        Total number of hrefs: 4635
Library: guidance
        Total number of results: 1582
        Total number of hrefs: 1259
Library: anthropic
        Total number of results: 1657
        Total number of hrefs: 1297
Library: llamaindex
        Total number of results: 116
        Total number of hrefs: 122
Library: langchain
        Total number of results: 55114
        Total number of hrefs: 42374
Library: openai
        Total number of results: 77702
        Total number of hrefs: 55214

Total number of files DOWNLOADED: 93134
```

In total, there are 104,901 hrefs. However, we have only 93134 files downloaded. Possible causes:
-  Duplicate hrefs since some files import more than one library
- 404 error, some files removed from github soon after it was detected by our script

There are 157k results on github with the following search query: 
```
"import guidance" OR "from guidance" OR "import openai" OR "from openai" OR "import langchain" OR "from langchain" OR  "import anthropic" OR "from anthropic" OR "import llamaindex" OR "from llamaindex" OR "import cohere" OR "from cohere" language:Python -repo:pisterlabs/prompt-linter
```
Since we now have 93k files, the dataset contains ~59% of open source file count.