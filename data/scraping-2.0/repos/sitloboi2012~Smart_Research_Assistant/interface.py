import streamlit as st
import pandas as pd
from question_generator import QuestionGenerator
from web_searcher import search_paper
from langchain.chains import LLMChain
#from constant import LLM_MODEL_4_SUMMARIZE
#from summarizer import SUMMARIZE_PROMPT
import asyncio


st.title("Personal Research Assistant :male-scientist:")
st.text("Hi, mình là Huy Mo 👨 - trợ lý ảo của bạn.")
st.text("Mình sẽ giúp bạn tìm kiếm các bài báo liên quan đến chủ đề mà bạn quan tâm.")

question_generator = QuestionGenerator()
#SUMMARIZE_CHAIN = LLMChain(llm = LLM_MODEL_4_SUMMARIZE, prompt = SUMMARIZE_PROMPT)
status = None

def generate_question(topic: str, description: str):
    output = question_generator.generate_question(topic, description)
    #final_response = question_generator.filter_result(topic, description, output)
    return output.lines

#async def summarize_abstract_async(abstract: str, title: str, study_field: str):
#        response = await SUMMARIZE_CHAIN.acall({"abstract": abstract, "title": title, "study_field": study_field})
#        return response["text"]

#async def process_summarize_abstract(list_of_result):
#    tasks = []
#    for i in list_of_result:
#        task = summarize_abstract_async(i["abstract"], i["title"], i["fieldsOfStudy"])
#        tasks.append(task)
#    return await asyncio.gather(*tasks)

def parsing_api_result(response_json):
    list_of_result = response_json["data"]
    #print(len(list_of_result))

    list_of_paper_id = [i["paperId"] for i in list_of_result]
    list_of_year = [str(i["year"]) for i in list_of_result]
    list_of_title = [i["title"] for i in list_of_result]
    list_of_abstract = [i["abstract"] for i in list_of_result]
    list_of_url = [i["url"] for i in list_of_result]

    #list_of_field_study = [i["fieldsOfStudy"] for i in list_of_result]
    list_of_field_study = [[field["category"] for field in i["s2FieldsOfStudy"]] for i in list_of_result]

    list_of_publication_date = [i["publicationDate"] for i in list_of_result]
    list_of_authors = [[author["name"] for author in i["authors"]] for i in list_of_result]
    list_of_count_authors = [len(i) for i in list_of_authors]
        
    list_of_references = [" || ".join([paper["title"] for paper in i["references"]]) for i in list_of_result]
    list_of_citation = [" || ".join([cite["title"] for cite in i["citations"]]) for i in list_of_result]
    list_of_references_count = [i["referenceCount"] for i in list_of_result]
    list_of_citation_count = [i["citationCount"] for i in list_of_result]
    
    list_of_bibtext_paper_citation = [i["citationStyles"]["bibtex"] for i in list_of_result]
    

    #list_of_summarize = asyncio.run(process_summarize_abstract(list_of_result))

    return {
        "paper_id": list_of_paper_id,
        "title": list_of_title,
        "abstract": list_of_abstract,
        "url": list_of_url,
        "field_study": list_of_field_study,
        "citation_count": list_of_citation_count,
        "references_count": list_of_references_count,
        "publication_date": list_of_publication_date,
        "authors": list_of_authors,
        "authors_count": list_of_count_authors,
        "year": list_of_year,
        "references": list_of_references,
        "citation": list_of_citation,
        "bibtext_paper_citation": list_of_bibtext_paper_citation,
        "raw_result": list_of_result
    }

def make_clickable(link, title):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">{title}</a>'

def convert_to_hyperlink(dataframe):
    clean_url = []
    for url, title in zip(dataframe["url"].tolist(), dataframe["title"].tolist()):
        if i is None:
            clean_url.append(None)
        else:
            clean_url.append(make_clickable(url, title))
    
    dataframe["url"] = clean_url
    return dataframe

import contextlib
with st.sidebar:
    form = st.form("Topic and Description Info Form")
    topic = form.text_area("Topic", value="XR in Marketing and Business", key="topic")
    description = form.text_area("Description", value="Unleashing the Metaverse: Extended Reality (XR) in Marketing", key="description")
    related_field = form.multiselect(label = "Field of study", options = ["Business","Economics","Education","Linguistics","Engineering","Political Science","Sociology","Computer Science","Psychology"], key="related_field", default = ["Business","Economics","Education","Linguistics","Engineering","Political Science","Sociology","Computer Science","Psychology"])
    submmited = form.form_submit_button(label = 'Start finding related papers 🔎')

if submmited:
    current_total = 0
    result = {"paper_id": [], "title": [],
              "abstract": [], "url": [],
              "field_study": [], "publication_date": [], 
              "citation_count": [], "references_count": [], 
              "authors": [],  "authors_count": [], "year": [], 
              "references": [], "citation": [],
              "bibtext_paper_citation": []}


    status = st.status("Finding related papers...", expanded=True)

    status.write("Generating list of keywords...")
    keyword_list = generate_question(topic, description)
    st.markdown("""List of keyword that has been used to find the papers: """)
    for i in keyword_list:
        st.markdown("- " + i)


    status.write("Crawling related papers...")
    progress_text = "Đợi xíu đi kiếm tài liệu cho bạn nè 🏃‍♂️"
    api_bar = st.progress(0, text=progress_text)
    current_progress = 0
    raw_result = []
    for index in range(len(keyword_list)):
        search_result = search_paper(keyword_list[index], ",".join(related_field))
        with contextlib.suppress(KeyError):
            parse_dict = parsing_api_result(search_result)
            current_total += search_result['total']
            result["paper_id"].extend(parse_dict["paper_id"])
            result["title"].extend(parse_dict["title"])
            result["abstract"].extend(parse_dict["abstract"])
            result["url"].extend(parse_dict["url"])
            result["field_study"].extend(parse_dict["field_study"])
            result["publication_date"].extend(parse_dict["publication_date"])
            result["authors"].extend(parse_dict["authors"])
            result["authors_count"].extend(parse_dict["authors_count"])
            result["year"].extend(parse_dict["year"])
            result["references"].extend(parse_dict["references"])
            result["references_count"].extend(parse_dict["references_count"])
            result["citation"].extend(parse_dict["citation"])
            result["citation_count"].extend(parse_dict["citation_count"])
            result["bibtext_paper_citation"].extend(parse_dict["bibtext_paper_citation"])
            raw_result.append(parse_dict["raw_result"])
        api_bar.progress(current_progress + 60, text=progress_text)

    st.markdown(f"Found __{current_total}__ papers related to the topic __{topic}__")
    api_bar.empty()
    status.write("Polishing the result...")



    #progress_text = "Đi summarize document 🏃‍♂️"
    #gen_bar = st.progress(0, text=progress_text)
    #for i in raw_result:
    #    result["summary_paper"].extend(asyncio.run(process_summarize_abstract(i)))
    #    gen_bar.progress(current_progress + 60, text=progress_text)
    #gen_bar.empty()


    status.update(label="Kiếm xong ùi check thử xem ạ 👏", state="complete", expanded=True)
    result_df = pd.DataFrame(result)
    #st.dataframe(result_df, use_container_width=True, column_config={"url": st.column_config.LinkColumn("URL to website")})
    st.data_editor(result_df, use_container_width=True, num_rows="dynamic", column_config={"url": st.column_config.LinkColumn("URL to website")}, hide_index=True)
    
    
    
    
    
    
    
    
    

