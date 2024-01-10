import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import scholarly
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import wikipediaapi
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

import nltk
nltk.download('punkt')

# Retrieve the OpenAI API key from the secrets.toml file
openai_api_key = st.secrets['openai']["apikey"]

# Password Protection

# Stream Handler for LLM generation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

class WikipediaAPIWrapper:

    def __init__(self, lang='en'):
        user_agent = "My User Agent 1.0"
        self.wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)

    def get_summary(self, topic):
        try:
            page = self.wiki.page(topic)
            if not page.exists():
                return None
            return page.summary[0:500]  # Limit the summary to the first 500 characters
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets['app_secret']["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    # st.write("Here goes your normal Streamlit app...")
    # App framework
    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        # Check the data types of the DataFrame
        #st.write(df.dtypes)

        # Inspect the first few rows of the DataFrame
        #st.write(df.head())


        # Check the unique values in 'product' and 'flow' columns
        #st.write(df['product'].unique())
        #st.write(df['flow'].unique())
        # Convert 'product' and 'flow' columns to lower case
        df['product'] = df['product'].str.lower()
        df['flow'] = df['flow'].str.lower()

        # Define the subsets
        subsets = [
            ['solar', 'wind', 'hydro'],
            ['natural gas: unabated', 'natural gas: with cccus'],
            ['coal: unabated', 'coal: with cccus'],
            ['oil', 'modern bioenergy: gas', 'hydrogen']
        ]

        # Convert the subsets to lower case
        subsets = [[product.lower() for product in subset] for subset in subsets]

        # Now the rest of your code...

        # Define the color palettes
        color_palettes = [
            sns.color_palette('Set2'),
            sns.color_palette('Dark2'),
            sns.color_palette('Set1'),
            sns.color_palette('Paired')
        ]

        # Create the plots
        fig, axs = plt.subplots(len(subsets), 1, figsize=(14, 6*len(subsets)))

        for i, subset in enumerate(subsets):
            # Filter the dataset for the current subset of products and 'total energy supply' flow
            df_subset = df[df['product'].isin(subset) & (df['flow'] == 'total energy supply')]

            # Create a color dictionary for the current subset of products
            colors = {product: color for product, color in zip(subset, color_palettes[i])}

            # Iterate through the products, scenarios and create a line for each combination
            for product in subset:
                for scenario in df['scenario'].unique():
                    data = df_subset[(df_subset['product'] == product) & (df_subset['scenario'] == scenario)]
                    if not data.empty:  # Check if there is data for the combination
                        axs[i].plot(data['year'], data['value'], color=colors[product], label=f'{product} - {scenario}')

            axs[i].set_title('Total Energy Supply for Selected Products by Scenario')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Total Energy Supply (EJ)')
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[i].grid(True)

        plt.tight_layout()
        st.pyplot(fig)


    ## AI part comes in    

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    # App framework
    st.markdown("<h1 style='text-align: center;'>Energy Scenario Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Get AI-generated scenarios for the energy industry of 2035</h2>", unsafe_allow_html=True)
    # Create a 1:1:1 layout with 3 columns
    col1, col2, col3 = st.columns([1,1,1])

    # Display the image in the middle column
    with col2:
        st.image('Shell_logo.png')  # Adjust the width as needed

    # User prompt
    st.markdown("<div style='text-align: center;'>**Please enter the factor you want to know about the energy industry:**</div>", unsafe_allow_html=True)

    prompt = st.text_input('', key='prompt_input')
    placeholder = st.empty()

    # Prompt templates
    scenario_template = PromptTemplate(
        input_variables=['topic'],
        template=
        """
        As an AI with extensive knowledge, I'm interested in understanding more about global energy trends, specifically regarding {topic}. There are several aspects of this that I would like you to consider:
        1. Can you provide an analysis on the {topic}, based on you research from IEA, reports from IOC like BP, Shell and TOTAL, and Myriad Report, Wood Mackenzie, and the economics ? 
        2. Taking into account the current energy trends, what are the predictions for {topic} in the year 2035 and up until the end of the decade?
        3. Utilizing your predictive capabilities, can you provide an estimation of {topic} in the year 2035 and up until the end of the decade?
        4. Can you identify any significant shifts in industry trends related to {topic} for the next decade?
        5. Given the present data and trends, how do you foresee the change in {topic} in the year 2035 and up until the end of the decade?
        Could you integrate your responses to these varied queries to form a comprehensive understanding of  
        a future scenario you imagined of {topic} for the energy industry in the year 2035 and up until the end of the decade?"
        Write the result in a list of key factors and trends on {topic} shaping the industry
        and how the industry might respond to these changes regarding {topic} .
        On top of these points, can you separately list what role might non-traditional players like tech companies, startups, and investment banks play in the {topic} by 2035?
        Write the result in a perfect markdown format, make the format beautiful.
        Based on the suggested scenarios and the documents found and synthesized, here is a future forecast report for Topic: {topic}"""
    )



    wiki_prompt_template = PromptTemplate(
        input_variables=['topic'],
        template="""Based on Research from Wikipedia:\n{topic}
        There are several aspects of this that I would like you to consider:
        1. Can you provide an analysis on the {topic}? 
        2. Taking into account the current energy trends, what are the predictions for {topic} in the year 2035 and up until the end of the decade?
        3. Utilizing your predictive capabilities, can you provide an estimation of {topic} in the year 2035 and up until the end of the decade?
        4. Can you identify any significant shifts in industry trends related to {topic} for the next decade?
        5. Given the present data and trends, how do you foresee the change in {topic} in the year 2035 and up until the end of the decade?
        Could you integrate your responses to these varied queries to form a comprehensive understanding of  
        a future scenario you imagined of {topic} for the energy industry in the year 2035 and up until the end of the decade?"
        Write the result in a list of key factors and trends on {topic} shaping the industry
        and how the industry might respond to these changes regarding {topic} .
        On top of these points, can you separately list what role might non-traditional players like tech companies, startups, and investment banks play in the {topic} by 2035?
        Also list any valid URLs that you get information from to inform your scenario.
        write the result in a perfect markdown format, use rich html to make the format beautiful.
        Include clickable citations as references, cite these Research Papers in APA citation format.
     ."""
    )

    scholar_prompt_template = PromptTemplate(
        input_variables=['topic'],
        template="""Based on Research from Google Scholar:\n{topic}
        There are several aspects of this that I would like you to consider:
        1. Can you provide an analysis on the {topic}? 
        2. Taking into account the current energy trends, what are the predictions for {topic} in the year 2035 and up until the end of the decade?
        3. Utilizing your predictive capabilities, can you provide an estimation of {topic} in the year 2035 and up until the end of the decade?
        4. Can you identify any significant shifts in industry trends related to {topic} for the next decade?
        5. Given the present data and trends, how do you foresee the change in {topic} in the year 2035 and up until the end of the decade?
        Could you integrate your responses to these varied queries to form a comprehensive understanding of  
        a future scenario you imagined of {topic} for the energy industry in the year 2035 and up until the end of the decade?"
        Write the result in a list of key factors and trends on {topic} shaping the industry
        and how the industry might respond to these changes regarding {topic} .
        On top of these points, can you separately list what role might non-traditional players like tech companies, startups, and investment banks play in the {topic} by 2035?
        write the result in a perfect markdown format, use rich html to make the format beautiful.
        Include clickable citations as references, cite these Research Papers in APA citation format."""
    )



    # Memory

    scenario_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    wiki_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    scholar_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    # LLM

    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method='write')
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key,temperature=0.9,max_tokens=1024,streaming=True,
                 callbacks=[stream_handler])


    wiki = WikipediaAPIWrapper()
    # Create the LLMChain model with the provided template
    scenario_chain = LLMChain(llm=llm, prompt=scenario_template, verbose=True, output_key='scenario',
                              memory=scenario_memory, callbacks=[stream_handler])

    wiki_chain = LLMChain(llm=llm, prompt=wiki_prompt_template, verbose=True, output_key='wiki_research', memory=wiki_memory, callbacks=[stream_handler])

    scholar_chain = LLMChain(llm=llm, prompt=scholar_prompt_template, verbose=True, output_key='scholar_research', memory=scholar_memory,callbacks=[stream_handler])



    # Fetch Wikipedia summary
    wiki_summary = wiki.get_summary(prompt)

    # Function to fetch Google Scholar research with source details

    def fetch_google_scholar_research(query, num_papers):
        try:

            num_papers=10
            search_query = scholarly.search_pubs_query(query)
            results = []
            for i, result in enumerate(search_query):
                if i >= num_papers:
                    break
                paper_title = result['bib']['title']
                paper_author = result['bib']['author']
                paper_url = result['url_scholarbib']
                paper_abstract = result['bib']['abstract']
                results.append({'title': paper_title, 'author': paper_author, 'url': paper_url, 'abstract': paper_abstract})
            return results
        except Exception as e:
            print(f"An error occurred while fetching Google Scholar research: {e}")
            return []

        # Function to summarize text using LexRank
    # Function to summarize text using LexRank
    def summarize_text(text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count=2)
        return ' '.join([str(sentence) for sentence in summary])
    scholar_research = fetch_google_scholar_research(prompt, 10)
    summarized_research = summarize_text(scholar_research)
    def display_answer(answer, sources):
        st.markdown('**AI-generated Scenario:**')
        st.info(answer)
        st.markdown('**Sources:**')

        if not sources:
            st.markdown('- No additional sources.')
        else:
            for source in sources:
                st.markdown(f"- **Title:** {source['title']}")
                if 'url' in source:
                    st.markdown(f"  **URL:** {source['url']}")
                if 'citation' in source:
                    st.markdown(f"  **Citation:** {source['citation']}")
                if 'summary' in source:
                    st.markdown(f"  **Summary:** {source['summary']}")
                if 'abstract' in source:
                    st.markdown(f"  **Abstract:** {source['abstract']}")
                st.markdown('---')



    # Process user prompt


    if placeholder.button("Run", type='primary'):

        if prompt:

            basic_result = scenario_chain.run(prompt)


            if basic_result:
                # Basic result
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box, display_method='write')

                basic_placeholder = st.empty()
                ai_placeholder = st.empty()
                basic_placeholder.success('AI-generated scenario with no additional sources.')
                display_answer(basic_result, [])
                #message(basic_result, is_user=False)
                with st.expander('Generative AI History'): 
                    st.info(scenario_memory.buffer)
            # Enrich with Wikipedia research
            wiki_placeholder = st.empty()

            if wiki_summary:
                wiki_prompt = wiki_prompt_template.format(topic=prompt)
                enriched_prompt_wiki = f"{wiki_prompt} Research from Wikipedia:\n{wiki_summary}"
                enriched_result_wiki = wiki_chain.run(enriched_prompt_wiki)

                if enriched_result_wiki:
                    display_answer(
                        enriched_result_wiki,
                        [{'title': 'Wikipedia Search Result', 'url': f"https://en.wikipedia.org/wiki/{prompt.replace(' ', '_')}",
                        'summary': wiki_summary}]
                    )
                    wiki_placeholder.success('AI-generated scenario enriched with Wikipedia research.')
                    with st.expander('Wikipedia research History'): 
                        st.info(wiki_memory.buffer)
            # Enrich with Google Scholar research and summarization
            scholar_placeholder = st.empty()

            scholar_research = fetch_google_scholar_research(prompt, 10)
            summarized_research = summarize_text(scholar_research)
            scholar_prompt = scholar_prompt_template.format(topic=prompt)
            enriched_prompt_scholar = f"{scholar_prompt}\n{summarized_research}"
            enriched_result_scholar = scholar_chain.run(enriched_prompt_scholar)

            if enriched_result_scholar:
                display_answer(enriched_result_scholar, [{'title': paper['title'], 'url': paper['url'], 'abstract': paper['abstract']} for paper in scholar_research])
                scholar_placeholder.success('AI-generated scenario enriched with Google Scholar research.')
                with st.expander('scholar History'): 
                    st.info(scholar_memory.buffer)





        # Add contextual hints around prompt box
        st.markdown('---')
        st.markdown('**💡 Contextual Hints:**')
        st.markdown('- Try asking about specific technologies, trends, or challenges in the energy industry.')
        st.markdown('- Ask how renewable energy sources may impact the future of the industry.')
        st.markdown('- Inquire about the role of government policies or emerging markets in shaping the industry.')
        st.button("Click me")




    
