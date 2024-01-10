from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatAnthropic
from read_creds import read_creds

def load_chain():
    read_creds()
    llm = ChatAnthropic(max_tokens_to_sample=10000)
    chain = ConversationChain(llm=llm)
    return chain


def load_topic_chain():
    read_creds()
    post_llm_prompt = """
    
    Human: <context>You are an assistant to an ad publisher. We want to embed ads into a Claude2 chatbot interface. To get the ads, we will send three searchable ad queries that you generate to SerpApi's google search results API. Then we will just scrape the ["shopping_results", "recipes_results","related_search_boxes", "organic_results"] of the google search response JSON to get the ["title", "link", "thumbnail"] fields.</context>
    
    <instructions>Please take the user query inside the <userQuery></userQuery> XML tags, and Claude2's response inside the <LLMResponse></LLMResponse> XML tags, and generate no more than three searchable ad queries. Generate queries based on the user's question in userQuery></userQuery> and specific things mentioned in the <LLMResponse></LLMResponse>. Return the queries inside of XML tags as 
    <adQueries>
        <query></query>
        <query></query>
        <query></query> 
    </adQueries>

    
    Do not include extra newlines between XML blocks like below
    <badExampleResponse>
    <adQueries>

    <query>redwood forest vacation packages</query>

    <query>redwood furniture buy online</query> 

    <query>redwood tree seedlings for sale</query>

    </adQueries>
    </badExampleResponse

    Please act as a XML code outputter. Do not add any additional context or introduction in your response; instead, make sure your entire response is parseable by XML.
    </instructions>

    <userQuery>{query}</userQuery>

    <LLMResponse>{response}</LLMResponse>
    
    Assistant:
    """

    chat_llm = ChatAnthropic()
    post_llm_prompttemplate = PromptTemplate(input_variables=['query', 'response'], template=post_llm_prompt)
    topic_chain = LLMChain(llm=chat_llm, prompt=post_llm_prompttemplate)

    return topic_chain

