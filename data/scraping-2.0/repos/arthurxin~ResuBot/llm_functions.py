from langchain.chat_models import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import PlaywrightURLLoader

def read_url(url):
    loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer"])
    data = loader.load()
    data =dict(data[0])
    page_content = data['page_content']
    return page_content

def summary_jd(api_key,url):
    url_content = read_url(url)
    my_key = api_key
    chat = ChatAnthropic(model="claude-2", anthropic_api_key = my_key, max_tokens_to_sample=1000)
    messages = [
        HumanMessage(
            content=f"""Try extract the job description, responsibility and requirements of this job, from the Given the content of a job recruitment website. Return me None if the content is not a job recruitment website. content:{url_content}"""
        )
    ]
    return dict(chat(messages))["content"]

def revise_resume(api_key,resume, jd, latex_templet):
    my_key = api_key
    chat = ChatAnthropic(model="claude-2", anthropic_api_key = my_key, max_tokens_to_sample=2000)
    messages = [
        HumanMessage(
            content=f"""revise the resume into a latex format based on the resume and job description below. 
All the information should based on the resume, only the format should be similar to the latex_format_templet.
Carefully read through the entire job description and highlight the key skills, qualifications, and requirements. These are the areas I want to emphasize in the resume.
Make sure the resume specifically uses words and phrases from the job description. 
In the resume's summary and experience sections, tailor your responsibilities and achievements to match what they are looking for. Quantify the accomplishments with facts/data when possible.
Try to make the resume  resume:{resume}. job description:{jd},  latex_format_templet:{latex_templet}"""
        )
    ]
    return dict(chat(messages))["content"]