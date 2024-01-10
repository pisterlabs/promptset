#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > ai_chatgpt_prompts.txt


# to install
pip install -r ai_chatgpt_prompts.txt

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_3_streamlit/

[file]
streamlit run 010_project_3_python_streamlit_app_chatgpt_api.py

# source
https://platform.openai.com/examples

pip install openai
pip install streamlit
pip install python-dotenv

"""

# os + dotenv to manage the OpenAI platform's licence
import os as os
from dotenv import load_dotenv

# streamlit
import streamlit as st

# OpenAI platform
import openai

# personal configuration
import config_values.values_conf as conf

# Load environment variables from .env file
load_dotenv()
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load the key
openai.api_key = OPENAI_API_KEY


### 1. VALUES ###
TEXT_TITLE_APP = conf.TEXT_TITLE_APP
TEXT_SUBHEADER_APP = conf.TEXT_SUBHEADER_APP
TEXT_WARNING = conf.TEXT_WARNING
TEXT_OUTPUT = conf.TEXT_OUTPUT

LABEL_EXPANDER = conf.LABEL_EXPANDER

# Usecase Title
MENU_SIDEBAR_USECASE_TITLE_OPTIONS = conf.MENU_SIDEBAR_USECASE_TITLE_OPTIONS

# Usecase Description
MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS = conf.MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS


TEXT_HELP_2 = conf.TEXT_HELP_2
GREAT_EXPENDER_TEXT_1 = conf.GREAT_EXPENDER_TEXT_1
GREAT_EXPENDER_TEXT_2 = conf.GREAT_EXPENDER_TEXT_2
GREAT_EXPENDER_TEXT_3 = conf.GREAT_EXPENDER_TEXT_3
TEXT_WARNING_REPORT = conf.TEXT_WARNING_REPORT
TEXT_WARNING_REPORT_HELP = conf.TEXT_WARNING_REPORT_HELP

TEXT_HELP_1 = conf.TEXT_HELP_1
TEXT_HELP_2 = conf.TEXT_HELP_2

### 2. USECASES VALUES ###

# CASE_6 :: 6. English to other languages
CASE_6_LANGUAGES_SELECTION = conf.CASE_6_LANGUAGES_SELECTION

TEXT_HELP_CASE_6_1 = conf.TEXT_HELP_CASE_6_1
TEXT_HELP_CASE_6_2 = conf.TEXT_HELP_CASE_6_2

# help for parameters
TEXT_HELP_TEMPERATURE_PARAMETER = conf.TEXT_HELP_TEMPERATURE_PARAMETER
TEXT_HELP_TOKENS_PARAMETER = conf.TEXT_HELP_TOKENS_PARAMETER


# CASE_3 :: 3. Summarize for a 2nd grader
TEXT_HELP_CASE_3_1 = conf.TEXT_HELP_CASE_3_1

# CASE_12 :: 12. Movie to Emoji (movie_to_emoji)
TEXT_HELP_CASE_12_1 = conf.TEXT_HELP_CASE_12_1

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = conf.TEXT_HELP_CASE_21_1

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = conf.TEXT_HELP_CASE_21_1

# CASE_22 :: 22. Python bug fixer
TEXT_HELP_CASE_22_1 = conf.TEXT_HELP_CASE_22_1

# CASE_22 :: 23. Spreadsheet creator
TEXT_HELP_CASE_23_1 = conf.TEXT_HELP_CASE_23_1

# CASE_39 :: 39. Notes to summary
TEXT_HELP_CASE_39_1 = conf.TEXT_HELP_CASE_39_1


# The code below is for the layout of the page
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

st.set_page_config(
    layout=layout, page_title=TEXT_TITLE_APP, page_icon="🔌"
)



class CaseChatGptDrawMainAppBody:

    def __init__(self):
        self.options = MENU_SIDEBAR_USECASE_TITLE_OPTIONS
        self.menu = st.sidebar.selectbox(
            "Menu options", self.options, help=TEXT_HELP_1)

        # cases 3, 6, 9, 10, 11, 12, 17, 21, 22, 23, 39
        
        self.selected_case = self.options.index(self.menu)
        # INTRO
        if self.selected_case == 0:

            self.summary_info = "case_0 summarize_info"
            self.prompt_input_model = None
            self.text_input_action_label = None
            self.output_button_label = None
            self.output_text_button_label = None
         
        # CASE_3
        # 3. Summarize for a 2nd grader
        elif self.selected_case == 3:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = 'Canzone napoletana (pronounced[kanˈtsoːne napoleˈtaːna.Neapolitan: canzona napulitana[kanˈdzoːnə napuliˈtɑːnə]), sometimes referred to as Neapolitan song, is a generic term for a traditional form of music sung in the Neapolitan language, ordinarily for the male voice singing solo, although well represented by female soloists as well, and expressed in familiar genres such as the love song and serenade. Many of the songs are about the nostalgic longing for Naples as it once was. The genre consists of a large body of composed popular music—such songs as "’O sole mio" and others. The Neapolitan song became a formal institution in the 1830s due to an annual song-writing competition for the Festival of Piedigrotta, dedicated to the Madonna of Piedigrotta, a well-known church in the Mergellina area of Naples.'
            self.text_input_action_label = "Summarize"
            self.output_text_button_label = "Text summarized"
            
        # CASE_6
        # 6. English to other languages
        elif self.selected_case == 6:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = 'How many children, do you have?'
            self.text_input_action_label = "Translate"
            self.output_text_button_label = "Text translated"
        
        # CASE_9
        # 9. Parse unstructured data
        elif self.selected_case == 9:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            # https://en.wikipedia.org/wiki/Tzatziki
            self.prompt_input_model = 'Tzatziki (Greek: τζατζίκι), also known as cacık (Turkish pronunciation: [dʒaˈdʒɯk]) or tarator, is a class of dip, soup, or sauce found in the cuisines of Southeastern Europe and the Middle East. It is made of salted strained yogurt or diluted yogurt mixed with cucumbers, garlic, salt, olive oil, sometimes with vinegar or lemon juice, and herbs such as dill, mint, parsley and thyme. It is served as a cold appetizer (mezze), a side dish, and as a sauce for souvlaki and gyros sandwiches and other foods.\n\n| Ingredients | Color | Flavor |'
            self.text_input_action_label = "Structure the data"
            self.output_text_button_label = "Data structured"
        
        # CASE_10
        # 10. Classification
        elif self.selected_case == 10:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = '\n\nApple, Facebook, Fedex\n\nNetflix, Nike, Gazprom\nCategory:'
            self.text_input_action_label = "Make classification"
            self.output_text_button_label = "Classify items into categories"
        
        # CASE_11
        # 11. Python to natural language
        elif self.selected_case == 11:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = '#Python\n\n def CallMeByMyName():\n\n MyName=input(\"Enter your name: \")\n\n for _ in range(50):\n\n print(MyName) \n\n# Explanation of what the code does\n\n#'
            self.text_input_action_label = "Explain this Python Code"
            self.output_text_button_label = "Python Code Explained"

        # CASE_12
        # 12. Movie to Emoji(movie_to_emoji)
        elif self.selected_case == 12:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = '\nThe godfather:'
            self.text_input_action_label = "Convert Movie to Emoji"
            self.output_text_button_label = "Emoji produced"


        # CASE_17
        # 17. Keywords
        elif self.selected_case == 17:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = 'What does the future hold for Wagner in Africa after the failed rebellion?\nRussian Foreign Minister Sergei Lavrov on Monday said that Wagner group troops would continue to operate in Mali and the Central African Republic following the rebellion led by their commander Yevgeny Prigozhin last weekend. Lavrov’s words came amid questions over the private militia’s role in Africa after more than five years of deployment to the continent.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018.\nMembers of the Close Protection Unit for Central African Republic President Touadera, composed of Russian private security company operatives from Sewa Security, are seen in Berengo on August 4, 2018. © Florent Vergnes, AFP\nThe Wagner group’s mutiny against Moscow last weekend has raised questions over the private militia’s presence in Africa.\nFor more than five years, thousands of Wagner group troops have deployed to Africa. In 2018 they were sent to support Central African Republic President Faustin-Archange Touadera as he grappled with an armed rebellion. In December 2021 they deployed to Mali in the wake of a coup and they have operated in Sudan since the end of 2017. Wagner troops have also been spotted in Libya and Mozambique.\nBut Lavrov vowed on Monday that the “events” of the last weekend would not impact the militia’s operations on the continent.\nThe Russian \"instructors\" and “private military contractors” in both Mali and the Central African Republic will continue their work, Lavrov said in an interview with Russia Today.\nPrigozhin’s rebellion will not change anything in Russia\'s ties with its allies, Lavrov added. \"There have been many calls(from foreign partners) to President(Vladimir) Putin ... to express their support, \" he said.\nNo African government has officially commented on the events of the past weekend. But according to Cyril Payen, senior reporter for FRANCE 24, Russia \"has undoubtedly become a slightly less reliable partner\" since Prigozhin’s rebellion.\"You can bet that people in Bangui and Bamako are wondering what the future holds, \" Payen added.\"The Malian state is now engaged in a double partnership, with the Russian state – the Putin camp – and with the Wagner group – the Prigozhin camp. Until now, this did not make much difference, but that could change if the two camps don’t reconcile in the long term, \" said lawyer and political scientist Oumar Berté in an interview with FRANCE 24’s sister radio station Radio France Internationale.\nOverlapping interests\nHowever, a senior official in the Central African Republic’s presidency told AFP that Russia will continue to operate in the central African country, with or without Wagner.\"The Central African Republic signed (in 2018, editor\'s note) a defence agreement with the Russian Federation, not with Wagner, \" said Fidele Gouandjika, special adviser to President Touadera.\"Russia has subcontracted with Wagner. If Russia no longer agrees with Wagner, then it will send us a new contingent.\"\n\nA bridgehead for Russian ambitions on the continent, the Central African Republic is particularly dependent on the Russian militia, whose men even work as private protection officers for Touadera.\n\nSome 1, 500 Wagner troops have been deployed to Mali since 2021. The paramilitary group has developed close ties with the junta in power, helping to train soldiers as well as taking part in operations to combat terrorist groups.\n\nPrigozhin’s men have also been seen in Libya, Sudan and Mozambique. Since the Wagner\'s group arrival in Africa, the UN, international NGOs and French authorities have regularly accused the paramilitary group of committing abuse and crimes against civilians.\n\nWagner always uses the same strategy every time it advances: disinformation campaigns(based on rejecting former colonial powers) and an offer of security in exchange for the exploitation of natural resources to supply Prigozhin’s war chest and serve the Kremlin’s interests.\n\nIn Sudan, the partnership between Wagner and the Rapid Support Forces(RSF), led by the junta’s number two, General Mohammed Hamdan Daglo, has enabled the paramilitary group to profit from illegal gold trafficking. It has also enabled them to organise the transport of the metal straight into the coffers of the Russian state, helping to swell its gold reserves and circumvent Western sanctions.'
            self.text_input_action_label = "Extract"
            self.output_text_button_label = "Keywords Extracted"

        # CASE_21
        # 21. TL;DR summarization
        elif self.selected_case == 21:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei."
            self.text_input_action_label = "TL;DR Summarize"
            self.output_text_button_label = "TL;DR Text summarized"

        # CASE_22
        # 22. Python bug fixer
        elif self.selected_case == 22:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = '##### Fix bugs in the below function\n\n def my_func(name, place): \n echo(f\'Hello name}! Are you from {place}?\') \n my_func("Prisca", "Paris")'
            self.text_input_action_label = "Fix python code"
            self.output_text_button_label = "Code fixed"

        # CASE_23
        # 23. Spreadsheet creator
        elif self.selected_case == 23:
            
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = "A four-column spreadsheet with a random choice of 5 countries with their name in english, name in spanish, capital and their top-level domains:\n\nNAME (English) | NOMBRE (Spanish) | CAPITAL | TLD"
            self.text_input_action_label = "Generate spreadsheet"
            self.output_text_button_label = "Spreadsheet generated"
        
        # CASE_39
        # 39. Notes to summary
        elif self.selected_case == 39:
            self.summary_info = MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[self.selected_case]
            self.prompt_input_model = "Tom: Profits up 50%\nJane: New servers are online\nKjel: Need more time to fix software\nJane: Happy to help\nParkman: Beta testing almost done"
            self.text_input_action_label = "Convert notes to summary"
            self.output_text_button_label = "Meeting Notes summarized"
        
        # Set default case
        else:
            self.summary_info = ""

    def DrawMainAppBody(self, text_input, selected_case):

        # cases 3, 6, 9, 10, 11, 12, 17, 21, 22, 23, 39
        
        # CASE_3
        # 3. Summarize for a 2nd grader
        if self.selected_case == 3:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"Summarize this for a second-grade student:\n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        # CASE_6
        # 6. English to other languages
        elif self.selected_case == 6:
            # debug
            # st.write(selected_case)
            
            # add languages
            self.languages = CASE_6_LANGUAGES_SELECTION
            self.selected_languages = st.multiselect(
                "Select Language", self.languages, help=TEXT_HELP_CASE_6_1)
            
            def extract_text(self, selected_languages):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"Translate into: {selected_languages} this text: \n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        # CASE_9
        # 9. Parse unstructured data
        elif self.selected_case == 9:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"A table summarizing the ingredients from: \n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        # CASE_10
        # 10. Classification
        elif self.selected_case == 10:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"The following is a list of companies and the categories they fall into: \n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0                    
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        # CASE_11
        # 11. Python to natural language
        elif self.selected_case == 11:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"Explanation of what the code does:{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["\n"]
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        
        # CASE_12
        # 12. Movie to Emoji(movie_to_emoji)
        elif self.selected_case == 12:
            # debug
            # st.write(selected_case)
            

            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"Convert movie titles into emoji.\n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["\n"]
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        # CASE_17
        # 17. Keywords
        elif self.selected_case == 17:
            # debug
            # st.write(selected_case)
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"Extract keywords from this text:\n\n{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.0
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        
        # CASE_21
        # 21. TL;DR summarization
        elif self.selected_case == 21:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"{text_input}\n\nTl;dr"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=1
                )
                return response.choices[0].text
                # debug
                # st.write(response)
            
        # CASE_22
        # 22. Python bug fixer
        elif self.selected_case == 22:
            # debug
            # st.write(selected_case)
            
            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=["###"]
                )
                return response.choices[0].text
                # debug
                # st.write(response)

        elif self.selected_case == 23 or self.selected_case == 39:
            # debug
            # st.write(selected_case)

            def extract_text(self):
                # Define function to summarize text using OpenAI Codex
                # Change to the desired OpenAI model
                model_engine = "text-davinci-003"
                # caution with the prompt
                prompt = f"{text_input}"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].text
                # debug
                # st.write(response)
        # CASE_6 :: 
        # 6. English to other languages
        if self.selected_case == 6:
            # debug
            # st.write(selected_case)
            st.markdown('**Set parameters:**')
            # Temperature and token slider
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help=TEXT_HELP_TEMPERATURE_PARAMETER, key="translate_temperature"
            )
            tokens = st.slider(
                "Tokens",
                min_value=60,
                max_value=2048,
                value=60,
                step=64,
                help=TEXT_HELP_TOKENS_PARAMETER, key="translate_tokens"
            )

            # Define Streamlit app behavior
            if st.button(f'{self.text_input_action_label}', key="translate_action"):
                st.write('')
                # debug
                # st.warning(f'temperature :: {temperature}')
                # st.warning(f'tokens :: {tokens}')
                # st.warning(self.selected_languages)

            self.output_text = extract_text(
                self.text_input, self.selected_languages)
            st.text_area(f'{self.output_text_button_label}', self.output_text)
        
        # 3, 6, 9, 10, 11, 12, 17, 21, 22, 23, 39
        elif self.selected_case == 3 or self.selected_case == 9 or self.selected_case == 10 or self.selected_case == 11 or self.selected_case == 12 or self.selected_case == 17 or self.selected_case == 21 or self.selected_case == 22 or self.selected_case == 23 or self.selected_case == 39:

            st.markdown('**Set parameters:**')
            # Temperature and token slider
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help=TEXT_HELP_TEMPERATURE_PARAMETER
            )
            tokens = st.slider(
                "Tokens",
                min_value=60,
                max_value=2048,
                value=60,
                step=64,
                help=TEXT_HELP_TOKENS_PARAMETER
            )
            # Define Streamlit app behavior
            if st.button(f'{self.text_input_action_label}'):
                st.write('')
                # debug
                # st.warning(f'temperature :: {temperature}')
                # st.warning(f'tokens :: {tokens}')
                self.output_text = extract_text(self.text_input)
                st.text_area(
                    f'{self.output_text_button_label}', self.output_text)
            
    def greatExpender(self, label, text):
        with st.expander(label):
            # st.info(f"{text}")
            st.success(self.summary_info, icon="✅")
            
    # Version sidebar
    def detectVersion(self):
        st.sidebar.markdown('* * *')
        st.sidebar.markdown('**VERSIONS**')
        st.sidebar.write("streamlit ::", st.__version__)

    # Main app design
    def doApp(self):

        st.header(f'{TEXT_TITLE_APP}')
        st.markdown(f'{TEXT_SUBHEADER_APP}')
        st.warning(f'{TEXT_WARNING}')

        selected_case = self.options.index(self.menu)
    
        for i, option in enumerate(self.options):
            if i == selected_case:
                st.markdown(f'### {option}')
            
                if self.summary_info:
                        self.greatExpender(
                        LABEL_EXPANDER, f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[selected_case]}")
                else:
                    st.write('')
                
                
                if self.prompt_input_model:
                    st.markdown('**Example:**')
                    st.code(self.prompt_input_model)
                    st.caption(
                        'For lazy folks, cut and paste the example :green[above]  into the textarea :red[below] to test it right away :sunglasses:')
                else:
                    st.write('')
                
                if self.text_input_action_label:
                    self.text_input = st.text_area(
                        f"{self.text_input_action_label}", help=TEXT_HELP_CASE_23_1)
                    
                    self.DrawMainAppBody(self.text_input, self.selected_case)
  
                else:
                    st.write('')
                
    # Do the app
    def run(self):
        self.doApp()
        self.detectVersion()


# Instance the class
app = CaseChatGptDrawMainAppBody()
app.run()



