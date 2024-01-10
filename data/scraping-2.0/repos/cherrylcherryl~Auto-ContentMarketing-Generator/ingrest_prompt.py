from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from apikey import load_env
import os

OPENAI_API_KEY, SERPER_API_KEY = load_env()
# import nest_asyncio
# nest_asyncio.apply()


MARKET_ANALYSIS_SITES = [
    'https://github.com/f/awesome-chatgpt-prompts',
    'https://www.greataiprompts.com/prompts/best-system-prompts-for-chatgpt/',
    'https://stackdiary.com/chatgpt/role-based-prompts/',
    'https://clickup.com/templates/ai-prompts/market-research-and-analysis',
    'https://snackprompt.com/topic/marketing/',
    'https://snackprompt.com/prompt/in-depth-market-research-data-analysis',
    'https://snackprompt.com/prompt/market-research',
    'https://snackprompt.com/prompt/in-depth-market-research-insights',
    'https://writesonic.com/blog/chatgpt-prompts/'
]

COMPETITOR_ASSESSMENTS_SITES = [
    'https://github.com/f/awesome-chatgpt-prompts',
    'https://www.greataiprompts.com/prompts/best-system-prompts-for-chatgpt/',
    'https://stackdiary.com/chatgpt/role-based-prompts/',
    'https://snackprompt.com/topic/marketing/',
    'https://resources.usemagnetiq.com/how-to-make-a-competitor-analysis-simply-with-chat-gpt/',
    'https://sproutsocial.com/insights/social-media-competitive-analysis/',
    'https://hero.page/samir/ai-prompts-for-saas-startups-jobs-prompt-library/competitor-analysis-content-gap-assessment',
    'https://writesonic.com/blog/chatgpt-prompts/',
    'https://www.linkedin.com/pulse/chatgpt-prompts-sales-paul-gentile'
]

UNIQUE_SELLING_POINT_SITES = [
    'https://github.com/f/awesome-chatgpt-prompts',
    'https://www.greataiprompts.com/prompts/best-system-prompts-for-chatgpt/',
    'https://stackdiary.com/chatgpt/role-based-prompts/',
    'https://snackprompt.com/topic/marketing/',
    'https://wgmimedia.com/chatgpt-prompts-for-dropshipping/',
    'https://proaiprompt.com/blog/chatgpt-prompts-for-crafting-unique-usps/',
    'https://writesonic.com/blog/chatgpt-prompts/'
]

CONTENT_CREATION_SITES = [
    'https://github.com/f/awesome-chatgpt-prompts',
    'https://www.greataiprompts.com/prompts/best-system-prompts-for-chatgpt/',
    'https://stackdiary.com/chatgpt/role-based-prompts/',
    'https://snackprompt.com/topic/marketing/',
    'https://writesonic.com/blog/chatgpt-prompts/',
    'https://snackprompt.com/prompt/unleash-your-creativity-content-idea-generation',
    'https://snackprompt.com/prompt/unleashing-engaging-podcast-content-with-chatgpt',
    'https://snackprompt.com/prompt/powerful-breaking-news-unique-content-generation',
    'https://snackprompt.com/prompt/seo-content-master',
    'https://snackprompt.com/prompt/mastering-tiktok-content-creation',
    'https://snackprompt.com/prompt/optimize-content-creator-pro-3',
    'https://snackprompt.com/prompt/content-creation-blueprint',
    'https://snackprompt.com/prompt/mastering-sales-enablement-content-creation',
    'https://snackprompt.com/prompt/mastering-blog-content-creation-with-chatgpt',
    'https://snackprompt.com/prompt/content-creation',

]

def add_documents(
        loader : WebBaseLoader, 
        instance : Chroma
    ):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators= ["\n\n", "\n", ".", ";", ",", " ", ""])
    texts = text_splitter.split_documents(documents)
    instance.add_documents(texts)

def ingrest_data(
        target_site : list[str], 
        embeddings : OpenAIEmbeddings, 
        path : str = 'data/integrated/market_analysis'
    ) -> None:
    instance = Chroma(embedding_function=embeddings, persist_directory=path)
    loader = WebBaseLoader(target_site)
    if loader:
        add_documents(loader, instance)
    instance.persist()



if __name__ == '__main__':

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
   
    ingrest_data(MARKET_ANALYSIS_SITES, embeddings, path='data/integrated/market_analysis')
    print("Integrated Market Analysis")
    ingrest_data(COMPETITOR_ASSESSMENTS_SITES, embeddings, path='data/integrated/competitor_assessments')
    print("Competitor Assessments Analysis")
    ingrest_data(UNIQUE_SELLING_POINT_SITES, embeddings, path='data/integrated/unique_selling_point')
    print("Integrated Unique Selling Point Analysis")
    ingrest_data(CONTENT_CREATION_SITES, embeddings, path='data/integrated/content_creator')
    print("Integrated Content Creation")