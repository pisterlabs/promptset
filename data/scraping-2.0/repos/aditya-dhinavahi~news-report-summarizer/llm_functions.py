from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from newspaper import Article, Config

load_dotenv()

llm = ChatOpenAI(temperature=0, model = "gpt-3.5-turbo-0613")

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 15

def get_text_from_url(url):
                article = Article(url, config = config)
                article.download()
                article.parse()
                text = article.text
                # print(llm.get_num_tokens(text))
                # print(text[0:1000])
                return article.text
            
def get_text_chunks(text, chunk_size = 2000, chunk_overlap = 500):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500,
    length_function = len
)
    # chunks = text_splitter.split_text(text)
    chunks = text_splitter.create_documents([text])
    # print(chunks)
    return chunks

def get_text_summary(text_chunks):

    map_prompt = '''
    Write a summary the text below delimited by <>
    Rules - 
    - Keep the length of the summary to max 160 words.
    - Retain the name of person, designation and company in the summary.
    - Keep the summary in format <title>, <person interviewed, designation, company>: <text summary>

    Text to summarize - <{text}>
    '''
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt =  """
    Summarize the content delimited by <> into a single paragraph and output the content with schema <title>, <person interviewed, designation, company>: <text summary>.
    Example output is given below delimited by ///
    Rules - 
    - Output of the summary should in format <title>, <person interviewed>, <designation> - <company>: <text summary>
    - Keep the length of output summary to max 160 words.
    ///
    ‘Indian banking sector is set for a golden decade amid growing digicalisation’, Uday Kotak, MD - Kotak Bank: details that the 
    financial sector landscape in India is at an interesting juncture, having witnessed global banking challenges and the 
    importance of stability. Despite past turbulence, India's financial sector has emerged stronger, but sustained growth 
    while prioritizing stability and sustainability is crucial.  He believes policymakers should focus on capacity-building 
    and though public sector banks have shown recovery, a larger capacity is needed to support India's growing economy. 
    He details the importance of technology in banking's future emphasizes that banks must balance legacy technology 
    with innovation. He cautions that risks and returns should be adequately priced with PLI scheme as companies to 
    scale, supporting India Inc.'s growth.
    ///
    Text to check on - <{text}>
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm, chain_type = "map_reduce", 
                                        map_prompt=map_prompt_template, 
                                        combine_prompt=combine_prompt_template)
    
    summary = summary_chain.run(text_chunks)
    return summary

def get_text_summary_custom(summary_type, prompt_user_text, text_chunks):
    if summary_type == "simple":
        template = f"""{prompt_user_text} \n \n

        Text to summarize - """ + """{text}
        SUMMARY: 
        """
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template)

        summary_chain = load_summarize_chain(llm=llm, chain_type='stuff', prompt = prompt)

    elif summary_type == "topic-wise":
        example_output = '''
Intelligence vs. Smarts: The article highlights that while schools tend to value and teach intelligence, smarts, which include empathy and communication skills, are more likely to lead to success in life. It emphasizes that the most important decision in life, choosing a partner, requires smarts rather than intelligence.

Understanding how the world works: The article emphasizes the importance of understanding how the world actually works in practice rather than just relying on theoretical knowledge. It states that people are emotional and make decisions based on various factors, not just logic.

Buying bubbles: The article mentions George Soros's strategy of buying overvalued investments, which may seem irrational to intelligent people but makes sense to smart people who understand the psychology of investors and the likelihood of bubbles growing larger.

Baseball: The article suggests that baseball has become boring over the last 20 years because teams have become more intelligent and focused on data-driven strategies, leading to a crisis of falling interest and attendance.

Accepting different perspectives: The article highlights the importance of intelligence in understanding that people with different lived experiences will have different perspectives and that debates often involve people talking past each other. It emphasizes that accepting and understanding different viewpoints is crucial for success.

Tolerating different views: The article states that the "right" answer to most problems is subjective and depends on individual well-being and experiences. It emphasizes that moving forward and getting things done requires tolerating and working with views that may differ from one's own.

'''


        map_prompt_template = "You are a smart editorial assistant who has to Write a summary the article below. \
    The summary is meant to be read by analysts who are smart and intelligent people. \
    So write in a clear and concise manner, don't use verbose language. \
    No need to state the obvious by starting the sentence with 'This text...', \
    just get right to the point.Write topic-wise summaries for the text below. \
    You will be give chunk of text delimited by <>. Get key topic of the text and write summary for the topic.\
                        \
    Output should look like the form $topic$:$topic summary$ \
    Actual text: <{text}>\n \
                        "
        combine_prompt_template = f"""{prompt_user_text} \n \n
                                        
                                        Output should look like -
                                        <topic1>:<topic1-summary>
                                        <topic2>:<topic2-summary>
                                        .
                                        .
                                        .
                                        <topicn>:<topicn-summary>
                                        
                                        Here's sample output given delimited by double hash signs ##
                                        #{example_output}#

                                        
                                        Text to summarize - """ + """{text} \n

                                        TOPIC-WISE SUMMARY:
                                        """
        map_prompt = PromptTemplate(template = map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template = combine_prompt_template, input_variables=["text"])
        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', 
                                            map_prompt=map_prompt, 
                                            combine_prompt=combine_prompt)
        
    output = summary_chain.run(text_chunks)
    return output
        