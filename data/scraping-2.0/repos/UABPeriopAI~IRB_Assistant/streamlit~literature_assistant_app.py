import tempfile
from datetime import datetime

import openai
import pypandoc
from llm_utils.database import get_db_connection
from llm_utils.streamlit_common import apply_uab_font, hide_streamlit_branding
from llm_utils.text_format import convert_markdown_docx

import IRB_Assistant.data as irb_data
import IRB_Assistant.generate as irb_generate
import IRB_Assistant_config.app_config as lit_app_config
import IRB_Assistant_config.config as irb_assistant_config
import streamlit as st


def show_literature_page():
    # page metadata
    st.set_page_config(
        page_title="Literature Search",
        page_icon="📚",
    )
    # hide streamlit branding
    hide_streamlit_branding()

    # apply uab font
    apply_uab_font()

    # page content
    st.title("📚 Literature Search 🤖")
    st.markdown(
        """
    **Use generative AI to situate your research question in the context of existing literature.**

    Brought to you by the Anesthesiology Research Support, Informatics, and Data Science teams.

    _Not approved for use with PHI._

    All submissions are recorded for potential review by departmental and health system personnel.

    ---
    """
    )

    email = st.text_input(
        "Your email address (NCBI requires an email address be associated with automated PubMed searches.)",
        "emailid@uabmc.edu",
    )
    search_type_options = [
        "assess the novelty of my idea",
        "get a jump start on my literature search",
        "estimate the complexity and feasibility of my idea",
        "identify weaknesses or gaps in the literature that serve as the key support for a proposed NIH-style grant",
    ]

    query_type = st.radio(
        "Which of these best describes what you want help with?", search_type_options
    )
    research_q = st.text_area(
        "Enter your research question (or for a grant, your specific aims)",
        "Retrospectively, In adult patients undergoing surgery, how does the use of regional anesthesia techniques compare to general anesthesia in terms of postoperative pain management?",
    )

    if st.button("Evaluate"):
        cost = 0.0
        input_time = datetime.now()
        article_ids = []
        loop_counter = 0
        previous_query = ""
        while len(article_ids) < irb_assistant_config.MIN_ARTICLES and loop_counter < 6:
            with st.spinner("Generating pubmed search string."):
                search_string, response_meta = irb_generate.generate_search_string(
                    research_q, loop_counter, previous_query
                )
                cost += response_meta.total_cost
                previous_query = search_string
                loop_counter += 1
            st.write(f"**Searching Pubmed with the query:** _{search_string}_")
            with st.spinner("Searching Pubmed."):
                article_ids_new = irb_data.search_pubmed_articles(
                    search_string,
                    email,
                    max_results=irb_assistant_config.MAX_PUBMED_RESULTS,
                    streamlit_context=True,
                )
                article_ids = list(set().union(article_ids, article_ids_new))
        if len(article_ids) > 0:
            st.write("**Found the following articles:**")
            article_infos = ""
            bibliography = ""
            additional_articles = ""
            articles = irb_data.fetch_article_details(article_ids, streamlit_context=True)
            for i, article in enumerate(articles):
                try:
                    authors = irb_data.format_authors(
                        article["MedlineCitation"]["Article"]["AuthorList"]
                    )
                except KeyError:
                    authors = ""

                try:
                    title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                except KeyError:
                    title = ""

                try:
                    journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
                except KeyError:
                    journal = ""

                try:
                    pub_year = datetime.strptime(
                        article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"][
                            "Year"
                        ],
                        "%Y",
                    ).year
                except KeyError:
                    pub_year = ""

                try:
                    abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
                except KeyError:
                    abstract = ""
                apa_citation = irb_data.format_apa_citation(
                    article, str(article["MedlineCitation"]["PMID"])
                )
                reference = f"{apa_citation}\n\n"
                if i < irb_assistant_config.MIN_ARTICLES:
                    st.write(apa_citation)
                    article_info = f"Authors: {authors}\nTitle: {title}\nJournal: {journal}\nPublication Year: {pub_year}\nAbstract: {abstract}\nAPA Citation: {apa_citation}\n\n"
                    article_infos += article_info
                    bibliography += reference
                elif i == irb_assistant_config.MIN_ARTICLES:
                    st.write("**...and more, which will be included in the downloadable file.**")
                    additional_articles += reference
                else:
                    additional_articles += reference
            with st.spinner("Summarizing and evaluating novelty"):
                overall_introduction, response_meta = irb_generate.generate_overall_introduction(
                    research_q, article_infos, query_type
                )
            response_time = datetime.now()
            cost += response_meta.total_cost
            output_text = ""
            output_text = (
                overall_introduction
                + "\n\n"
                + "\n\n **Works considered in above text:**\n\n"
                + bibliography
                + "\n\n **Additional articles found:**\n\n"
                + additional_articles
            )
            docx_data = convert_markdown_docx(output_text)

            if docx_data:
                st.balloons()
                st.write("Note that once you hit download, this form will reset.")

                st.download_button(
                    label="Download Evaluation",
                    data=docx_data,
                    file_name="Literature_novelty_evaluation.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # correct MIME type for docx
                )
        else:
            st.write("No articles found")

        try:
            with get_db_connection(
                db_server=lit_app_config.DB_SERVER,
                db_name=lit_app_config.DB_NAME,
                db_user=lit_app_config.DB_USER,
                db_password=lit_app_config.DB_PASSWORD,
            ) as conn:
                # tempting to move this into llm_utils, but the query will be unique to each app.
                cursor = conn.cursor()
                query = """
                        INSERT INTO [dbo].[literature_helper] (
                            email_address, 
                            research_idea, 
                            purpose_request, 
                            citations, 
                            literature_summary, 
                            pubmed_query, 
                            input_time, 
                            response_time,
                            total_cost
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """

                cursor.execute(
                    query,
                    (
                        email,
                        research_q,
                        query_type,
                        bibliography + "\n\n" + additional_articles,
                        overall_introduction,
                        search_string,
                        input_time,
                        response_time,
                        cost,
                    ),
                )
            st.success(
                "To comply with a Health System Information Security request, submissions are recorded for potential review."
            )
        except Exception as e:
            st.error(
                "Something went wrong, and your submission was not recorded for review. Give the following message when asking for help."
            )
            st.error(e)


if __name__ == "__main__":
    show_literature_page()
