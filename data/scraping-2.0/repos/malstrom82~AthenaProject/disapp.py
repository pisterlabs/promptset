import streamlit as st
import joblib
import openai
import gdown

##################################
#import urllib.request
#import os

# For saved_model.pk3
#url1 = 'https://github.com/malstrom82/AthenaProject/releases/download/version1/saved_model.pk3'
#filename1 = url1.split('/')[-1]
#urllib.request.urlretrieve(url1, filename1)

# For saved_model.pk4
#url2 = 'https://github.com/malstrom82/AthenaProject/releases/download/version1/saved_model.pk4'
#filename2 = url2.split('/')[-1]
#urllib.request.urlretrieve(url2, filename2)
#################################
# Define a function to download and cache the models
#def load_models():
    # Define the URLs for your models
    #model_url1 = 'https://drive.google.com/file/d/1LaC4Kh-ANtqeBafUxYabsP_hBAvGPyQE'
    #model_url2 = 'https://drive.google.com/file/d/1FEaetS71MEWf59-jPyvkPH8So3ct2p7j'
    #model_url1 = 'https://github.com/malstrom82/AthenaProject/blob/main/saved_model.pk3'
    #model_url2 = 'https://github.com/malstrom82/AthenaProject/blob/main/saved_model.pk4'

    # Define filenames for caching
    #filename1 = 'model1.pk3'
    #filename2 = 'model2.pk4'
    
    # Check if the models already exist locally and if not, download them
    #if not os.path.exists(filename1):
    #    urllib.request.urlretrieve(model_url1, filename1)
    #if not os.path.exists(filename2):
    #    urllib.request.urlretrieve(model_url2, filename2)

    #print(f"Size of {filename1}: {os.path.getsize(filename1)} bytes")
    #print(f"Size of {filename2}: {os.path.getsize(filename2)} bytes")
    
    # Load your models here using joblib or other appropriate methods
    #model1 = joblib.load(filename1)
    #model2 = joblib.load(filename2)

    #return model1, model2

# Check if models are already loaded using SessionState
#if 'loaded_models' not in st.session_state:
# Load the models and store them in the session state
#    st.session_state.loaded_models = load_models()

# Get the loaded models from the session state
#model1, model2 = st.session_state.loaded_models

##################################
# Caching the download function ensures model files are only downloaded once
@st.cache_data(show_spinner=True)
def download_model(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

# Google Drive file IDs for your model files
model_file_id_1 = '1LaC4Kh-ANtqeBafUxYabsP_hBAvGPyQE'
model_file_id_2 = '1FEaetS71MEWf59-jPyvkPH8So3ct2p7j'

# Paths to save the downloaded models
model1 = 'saved_model.pk3'
model2 = 'saved_model.pk4'

# Download the model files
download_model(model_file_id_1, model1)
download_model(model_file_id_2, model2)

#################################
####################
# Set up OpenAI API key
# REDACTED

# Retrieve the OpenAI API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Set up OpenAI API key
openai.api_key = api_key
#####################

page = st.sidebar.radio("**Navigation**", ["Home", "Credibility Checker", "Disinformation Detector", "Legal Helper", "About"])
###############
#model_path_1 = "saved_model.pk3"
#model_path_2 = "saved_model.pk4"
###################
##########################################################################################################################################
##########################################################################################################################################
if page == "Home": 
    st.title("Athena-Disapp: Credibility assessment in your pocket")

    st.header("Tools & suggested workflow")

    # Create three columns for the layout
    col1, col2, col3 = st.columns(3)

    # Content for Column 1 - Fake News App
    with col1:
        st.write("**Step 1 - Credibility Checker**")
        st.info(""" A machine learning model trained on a large dataset predicts if an article is credible or not.
       """)

    # Content for Column 2 - Disinformation Checker
    with col2:
        st.write("**Step 2 - Disinformation Detector**")
        st.info("""
       An experimental tool that explores the possibility, using a machine learning model, to classify whether a non-credible article is misinformation or disinformation.""")

    # Content for Column 3 - EU Legal Document Chatbot
    with col3:
        st.write("**Step 3 - Legal Helper**")
        st.info("""
       Designed to provide better understanding of the EU legalization documents, such as the AI act or GDPR.""")

    st.header("FIMI and project")
    st.write("Developed by master-students at Gothenburg's University in collaboration with RISE, the Research Institutes of Sweden. This website aims to educate the user on Foreign information manipulation and interference (FIMI), misinformation, disinformation, the differences between them and a further look into the legislative frameworks adapted in the EU. Learn more about the project in the about section.")
    st.header("Glossary")
    st.write("**Foreign information manipulation and interference (FIMI)**")
    st.write("“A pattern of behavior in the information domain that threatens values, procedures and political processes. Such activity is manipulative (though usually not illegal), conducted in an intentional and coordinated manner, often in relation to other hybrid activities. It can be pursued by state or non-state actors and their proxies.”")
    st.write("[Learn more](https://www.eeas.europa.eu/eeas/tackling-disinformation-foreign-information-manipulation-interference_en)")
    
    st.write("**Misinformation**")
    st.write("“False or misleading content shared without intent to cause harm. However, its effects can still be harmful, e.g. when people share false information with friends and family in good faith.”")
    st.write("[Learn more](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52020DC0790&from=EN)")


    st.write("**Disinformation**")
    st.write("“False or misleading content that is created, presented and disseminated with an intention to deceive or secure economic or political gain and which may cause public harm. Disinformation does not include errors, satire and parody, or clearly identified partisan news and commentary.”")
    st.write("[Learn more](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52020DC0790&from=EN)")

    st.write("**All this sounds like “fake news”, why the new words?**")
    st.write(" As we try to follow [EuvsDisinfo](https://euvsdisinfo.eu/) we do not use “Fake news” as a concept due to the strong political connotations to the phrase and that it is woefully inaccurate to describe the complexity of the issues at stake. Instead, we choose the words 'credible' and 'non-credible' to describe a piece of news media that should not be trusted.")

    st.header("About")
    st.write("Here you can read more about the project, ATHENA, RISE our mastersprogram and our information if you have any questions.")
##########################################################################################################################################
##########################################################################################################################################
if page == "Credibility Checker":
    st.title("Article Credibility Analysis")
    st.write("This is an AI-powered credibility application. Developed by master-students at Gothenburg's University in collaboration with RISE, the Research Institutes of Sweden. Learn more about the project in the about section.")
    st.write("Paste the body of an article you want to check if it is credible or not. A machine learning (ML) classifier will predict if the article is credible or not. And the OpenAI large language model will make further analysis and give you insights to fake news, misinformation and applicable legal frameworks.")
    st.write("For the analysis of the text, ChatGPT was utilized. Customizing prompts enabled the extraction of specific analytical insights from the verdict. The machine learning (ML) verdict is integrated within the prompt messages, holding the highest value. ChatGPT operates based on information available up to January 2022.")
    st.write("Option 1: Paste an article or text you want to analyze, and press 'Analyze article'.\n\nOption 2: Together with your article, add the news outlet/source and author of the article, to get a deeper analysis.\n\nOption 3: If you only want a quick check on an author or source, enter them below and press 'Check only source' or 'Check only author'.")
    #st.write("For now, only enter text into 1 text-box, and press the relevant analyze-button. Functionality will be added with a later update.")
    
    left_column, right_column = st.columns([1,2])
    #st.set_option('deprecation.showPyplotGlobalUse', False)     ## gör denna nåt?
    with left_column:
        st.image("textanalys.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
    with right_column:
        artikel_input = st.text_input("Paste your article here:", key="article_input", help="""**Examples of prompts for chat-GPT**:   
        - Your primary task is to analyze the given text, identifying indicators of potential disinformation or fake news. This analysis should span across linguistic cues, historical veracity, and more.  
        - Ensure responses are succinct and fact-driven. Your role is to guide EU decision-makers by analyzing for signs of disinformation within relevant legal frameworks.""")

        col1, col2 = st.columns(2)
        source_input = col1.text_input("For deeper analysis, paste the news outlet or source here (optional):", key="source_input")
        send_source_button = col2.button("Check only source")

        col3, col4 = st.columns(2)
        author_input = col3.text_input("For deeper analysis, paste the author name here (optional):", key="author_input")
        send_author_button = col4.button("Check only author")
        
        send_request = st.button("Analyze article")

        # Handle "Send Source" button press
        if send_source_button:
            messages = [
                # ... messages tailored for source analysis ...
                {"role": "system", "content": f"Your role is to nalyze the provided source/news outlet: {source_input} for credibility, in terms of previous articles and publications, and connections to fake news or disinformation."},
                {"role": "system", "content": "Use your knowledge about the source, and give a description of the source, and in what way it can be tied to fake news or disinformation, or if the source can be seen as credible. Motivate your answer, and make sure to mention the source by name in your response."},
                {"role": "system", "content": "Ensure responses are succinct and fact-driven, max around 100 words."},
            ]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3
            )
            response = completion.choices[0].message.content
            right_column.write(response)
        
        # Handle "Send Author" button press
        if send_author_button:
            messages = [
                # ... messages tailored for author analysis ...
                {"role": "system", "content": f"Your role is to nalyze the provided author: {author_input} for credibility, in terms of previous articles, and connections to fake news or disinformation."},
                {"role": "system", "content": "Use your knowledge about the author, and give a description of the author, and in what way he/she can be tied to fake news or disinformation, or if the author can be seen as credible. Motivate your answer, and make sure to mention the authors name in your response."},
                {"role": "system", "content": "Ensure responses are succinct and fact-driven, max around 100 words."},
            ]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3
            )
            response = completion.choices[0].message.content
            right_column.write(response)
        
        # Handle main "Analyze" button press
        if send_request:
            # Retrieve article, source, and author input from the session state
            article_for_analysis = st.session_state.article_input
            source_for_analysis = st.session_state.source_input
            author_for_analysis = st.session_state.author_input

            # Debug output to verify inputs
            #st.write("Article:", article_for_analysis)
            #st.write("Source:", source_for_analysis)
            #st.write("Author:", author_for_analysis)
            
            user_input = artikel_input
            ### cash kod ####
            #pipeline = model1
            #user_input_bow = pipeline.named_steps['bow'].transform([user_input])
            #proba_real = pipeline.predict_proba([user_input])[0][0]
            ##############
            pipeline = joblib.load(model1)            ## modellen
            user_input_bow = pipeline.named_steps['bow'].transform([user_input])
            proba_real = pipeline.predict_proba([user_input])[0][0]  # adjusted the index
            # Check the probability range
            if 0.4 <= proba_real <= 0.6:
                st.write("Analysis not possible - no clear signs for or against this articles credibility. Human analysis needed")
                clverdict = "Credibility Analysis Not Possible."
            elif proba_real > 0.6:
                st.write("Important note: The model makes no analysis of author intent. It is also important to keep in mind that the models verdict is based on probabilities, and can not be used as the single source for judgement on an articles credibility.")
                clverdict = "Probably Credible."
            else:
                st.write("Important note: The ML model used makes no analysis of author intent. It is also important to keep in mind that the models verdict is based on probabilities, and can not be used as the single source for judgement on an articles credibility.")
                clverdict = "Probably Not Credible."

            #st.write("classifier raw verdict (for test):", clverdict) # debugging
            
            messages = [
                # ... messages for complete analysis ...
                {"role": "system",
                    "content": f"Your primary task is to analyze the given text ({article_for_analysis}), identifying indicators of potential disinformation or fake news. This analysis should span across linguistic cues, historical veracity, and more."
                },
                {"role": "system",
                    "content": "Ensure responses are succinct and fact-driven. Your role is to guide EU decision-makers by analyzing for signs of disinformation within relevant legal frameworks."
                },
                {"role": "system",
                    "content": "Structure your analysis under the following headers: 'Analytics:', 'Summary:', 'Legality:', 'Disclaimer:'."
                },
                {"role": "system",
                    "content": "In the 'Analytics' section, present each analytical tool, its analysis, and the verdict, all in a clear list."
                },
                ######## för test ###########################################
                #{"role": "system",
                #    "content": "You can simulate analytical results and scenarios. The goal is to assess the feasibility and effectiveness of the concept."
                #},
                {"role": "system",
                    "content": "For the analytics section, use the following headers and sub-headers: ML Classifier assessment, Source verification, Author verification, Textual analysis (sub: Fact checking & Sentiment analysis), Network analysis (sub: Propagation pattern analysis & Source cluster analysis), Linguistic analysis (Sub: Spelling and Grammar check & Writing style analysis), Bias & Agenda analysis, Historical crosscheck."
                },
                {"role": "system",
                    "content": "Use the following prompts for analysis:"
                },
                ######## classifiern ###################################################
                {"role": "system",
                    #"content": "ML classifier assessment: 'Based on a ML classifier (xx logistic regression, trained on yy database), this article is classified as most likely (placeholder).'. For now, make this verdict true/false, based on the majority of the other metrics used below. Leter use this as the verdict of the ML classifier in your later summarization of all metrics."
                    "content": f"ML classifier assessment: 'Based on a ML classifier (xx logistic regression, trained on yy database), this article is classified as {clverdict}.'" ####### få in variabeln här (obs dummy-kod)
                },
                {"role": "system",
                    "content": f"together with 'ML Classifier assessment', also type out the raw text in this variable: {clverdict}."
                },    
                #{"role": "system",
                #    "content": "Source Verification: 'Based on historical data, is this articles source a reputable source for accuracy and credibility?' Response should be: 'Source Verification: The source of this text is reputable/non-reputable based on historical data.', or 'No historical data on this source was found - credibility cannot be confirmed'."
                #},
                {"role": "system",
                    "content": f"If you receive an source {source_for_analysis}, make a binary verdict based on your knowledge, is this source generally seen as credible, or not credible? Use this verdict in the 'Source Verification' section of your response."
                },
                {"role": "system",
                    "content": f"If you receive a source name {source_for_analysis}, use this as the source in the 'Source Verification' below. If you receive a source {source_for_analysis}, name it and give a short description of it (maximum one sentence), pointing out how it is most likely credible/not credible. If you do not recieve a source, State 'No source found.'. '''If you receive a source, but you are unable to verify its credibility, state this and explain why you were not able to determine its credibility.'''"
                },
                #{"role": "system",
                #    "content": "Source Verification: 'Based on historical data, is this articles source a reputable source for accuracy and credibility?' Response: Provide a response using the provided text, and your knowledge, or 'No historical data on this source was found - credibility cannot be confirmed'."
                #},
                {"role": "system",
                    "content": "Source Credibility: Historically, how credible is the source in terms of journalistic integrity? use your source credibility verdict from above to say credible or not credible."
                },
                #{"role": "system",
                #    "content": "Author Credibility: 'Historically, how credible is the author in terms of journalistic integrity?' Response: 'Author Credibility: The author of the text is credible/non-credible based on past articles.', or 'No past articles tied to this author was found'."
                #},
                {"role": "system",
                    "content": "Author Credibility: Historically, how credible is the author in terms of journalistic integrity? use your author credibility verdict from above to say credible or not credible."
                },
                {"role": "system",
                    "content": f"If you receive an author {author_for_analysis}, make a binary verdict based on your knowledge, is this author generally seen as credible, or not credible? Use this verdict in the 'Author Credibility' section of your response."
                },
                {"role": "system",
                    "content": f"If you receive an author {author_for_analysis}, use this as the source in the 'Source Credibility' section below. If you receive an author {author_for_analysis}, name it and give a short description of it, pointing out how it is most likely credible/not credible. If you do not recieve an author, State 'No Author found.'. '''If you receive an author, but you are unable to verify its credibility, state this and explain why you were not able to determine its credibility.'''"
                },
                {"role": "system",
                    "content": "Author Credibility: 'Historically, how credible is the author in terms of journalistic integrity?' Response: Provide a response using the provided text and your knowledge, or 'No past articles tied to this author was found'."
                },
                {"role": "system",
                    "content": "Fact Check: 'Is the statements or facts in the provided text historically accurate?' Response: Provide a response using the provided text, and your knowledge."
                },
                {"role": "system",
                    "content": "Sentiment Analysis: 'Does the provided text [text] convey a neutral or biased sentiment?' Response: Provide a response using the provided text, and your knowledge."
                },
                {"role": "system",
                    "content": "Network Analysis: 'Historically, how have similar topics or claims propagated in credibility and source distribution?' Response: 'Network Analysis: Topics similar to this have [spread through reputable channels/been associated with misinformation campaigns].'"
                },
                {"role": "system",
                    "content": "Linguistic Analysis: 'Are there grammatical or spelling errors in this sentence: [sentence]?' Response: 'Linguistic Check: The text [has/doesn't have] significant linguistic errors.'"
                },
                {"role": "system",
                    "content": "Writing Style Analysis: 'Does the text [text] use sensationalist or emotive language indicating bias?' Response: 'Writing Style Check: The text [uses/doesn't use] sensationalist or emotive language.'"
                },
                {"role": "system",
                    "content": "Bias & Agenda Analysis: 'Does the text [text snippet] show indications of political or commercial bias?' Response: 'Bias Analysis: The text [shows/doesn't show] signs of [political/commercial] bias.'"
                },
                {"role": "system",
                    "content": "External Database Check: 'Up to 2022, have similar claims been identified as fake news: [claim or topic]?' Response: 'Database Check: This type of claim [has/hasn't] been identified in historical fake news records.'"
                },
                {"role": "system",
                    "content": "After analysis, aggregate the 'likely true' and 'likely false' verdicts for a summarized result. Example: 'After assessing all metrics, the content appears likely true/mostly false.'"
                },
                {"role": "system",
                    "content": "Always include a Summary-section (after Analysis, before Legality): 'Summary: After assessing all metrics, the content appears to be likely credible/most likely not credible.'. In the summary, analyse all the results from the analysis section, and combine the results into a final verdict. Always include either 'Based on the full analysis, the article is most likely **credible**' and provide your reasoning for this, or 'Based on the full analysis, the article is most likely **not credible**' with your motivation, or a neutral similar answer, if analysis did not provide strong fake/real info."
                },
                {"role": "system",
                    "content": "In the summary section, when you assess all analytic metrics and give your final summarized verdict of credible/not credible, give the result from the ML classifier the most weight. Use the other metrics to support the ML classifier, or to question it's accuracy, if all other metrics diverge from the classifier verdict in their verdicts."
                },
                {"role": "system",
                    "content": "For legality, reference these frameworks: GDPR, AI Act, NIS 2 Directive, 2019 Cybersecurity Act, e-Evidence Act, Digital Service Act, AI Liability Act. Include other relevant EU frameworks only when pertinent. if the text does not relate in any way to one or more legal frameworks, leave the Legality-section empty, with the text 'No relevant connections found to known EU legal frameworks.'."
                },
                {"role": "system",
                    "content": "Always end your response with a disclaimer-section: 'Disclaimer: Labeling an article as 'fake' or 'real'/credible or not credible is provisional due to inherent uncertainties. Consult multiple methods for any credibility assessment.'.  If deemed 'fake', mention the broader societal and individual implications of disinformation. For suspected disinformation, provide avenues for reporting this to relevant authorities."
                },
                {"role": "system",
                    "content": "If external sources are consulted or utilized, list them comprehensively at the end of your response. If none are used, state this."
                },
                {"role": "system",
                    "content": "Your response will be used in a streamlit app. Make sure you use the following formatting: In all your response, make these BOLD: Section headers, sub headers (from 'ML Classifier assessment' to 'External Database Check'), and the 'likely true' or 'likely fake' in the summary-section. Sub-headers with their text should be in pointed lists."
                },
                #{"role": "system",
                #    "content": "For all sections and prompts: If you dont know enough information to make an analysis or provide an answer, state this as 'Not enough information to provide an analysis'. In this case leave that analysis blank, with only this message."
                #},
                ############# FIXA!!
                {"role": "system",
                    "content": "If in your analysis you find signs of disinformation/fake news in the text (it has sensationalist language, biased or infactual content or some other sign), give examples from the text of where you see those signs, and why. Do this for each of the analytic tools above, if you find clear signs."
                },
            ]
            
            if source_input:
                messages.append({"role": "system", "content": f"This is the source/media outlet of the article you are analysing: {source_for_analysis}."})
                messages.append({"role": "system", "content": "for the source/media outlet verification, use your historical knowledge, to decide if this source/media outlet has previously posted credible content, or not."})
                #messages.append({"role": "system", "content": "Be sure to mention the source by name in your verdict."})           ### för test      
                #messages.append({"role": "system", "content": "Ensure responses are succinct and fact-driven."})
            
            if author_input:
                messages.append({"role": "system", "content": f"This is the author of the article you are analysing: {author_for_analysis}."})
                messages.append({"role": "system", "content": "for the author credibility analysis, use your historical knowledge, to decide if this author has previously written credible work in credible media, or not."})
                #messages.append({"role": "system", "content": "Be sure to mention the author by name in your verdict."})          ### för test  
                #messages.append({"role": "system", "content": "Ensure responses are succinct and fact-driven."})

            # Debug output to verify the complete messages list
            #st.write("Messages for OpenAI API:", messages)
        
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3
            )
        
            response = completion.choices[0].message.content
            right_column.write(response)
        
        #######################################################################
        
        #pipeline = joblib.load(model_path_1)
        #st.title("Article Credibility Analysis")
        # User input for article
        #user_input = st.text_area("Enter an article:")
        
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Add a "Send Question" button
        #if st.button("Check article"):
        #if user_input:
            #user_input_bow = pipeline.named_steps['bow'].transform([user_input])
        # Get the probability for the real news (labeled as '0')
            #proba_real = pipeline.predict_proba([user_input])[0][0]  # adjusted the index 
        
        # Check the probability range
            #if 0.4 <= proba_real <= 0.6:
                #st.write("Analysis not possible - no clear signs for or against this articles credibility. Human analysis needed")
            #    clverdict = "Unavailable"
            #elif proba_real > 0.6:
                #st.write("The provided article has been flagged as '**Probably Credible**'. The analysis is based on a Logistical Regression ML-model, trained on a database of known false and true news articles. \n\nThe model makes no analysis of author intent. It is also important to keep in mind that the models verdict is based on probabilities, and can not be used as the single source for judgement on an articles credibility.")
             #   clverdict = "Credible"
            #else:
                #st.write("The provided article has been flagged as '**Probably Not Credible**'. \n\nThe analysis is based on a Logistical Regression ML-model, trained on a database of known false and true news articles. \n\nThe model makes no analysis of author intent. It is also important to keep in mind that the models verdict is based on probabilities, and can not be used as the single source for judgement on an articles credibility.")
            #    clverdict = "Not Credible"
##########################################################################################################################################
##########################################################################################################################################
if page == 'Disinformation Detector':
    st.title("Disinformation classifier")
    st.write("Experimental ML classifier for classifying whether a non-credible article is disinformation or misinformation. A more nuanced dataset is necessary to obtain trustworthy results. Read more on the 'about' page.") 
    st.write("Paste a non-credible article in the text box and the classifier will predict if the article is misinformation or disinformation. This is an experimental classifier and the results should be treated as experimental, learn more about this model on our about page.")

    #pipeline = joblib.load(model_path_2)

    # User input for article
    user_input = st.text_area("Enter an article:")

    # Add a "Send Question" button
    if st.button("Check article"):
        if user_input:
            #pipeline = model2
            #user_input_bow = pipeline.named_steps['bow'].transform([user_input])
            #proba_real = pipeline.predict_proba([user_input])[0][0]  # adjusted the index
            
            pipeline = joblib.load(model2)            ## modellen
            user_input_bow = pipeline.named_steps['bow'].transform([user_input])
            ## Get the probability for the real news (labeled as '0')
            proba_real = pipeline.predict_proba([user_input])[0][0]  # adjusted the index

    # Check the probability range
            if 0.4 <= proba_real <= 0.6:
                st.write("Classification not possible - no clear signs of either Misinformation or Disinformation.")
            elif proba_real > 0.6:
                st.write("This article has been flagged as **Disinformation**. \n\nThis means that the classifier found high similarity between the article and other articles known to be disinformation. \n\nThe difference between disinformation and misinformation has to do with author intent. The classification of this article indicate a **high** probability of author malintent.\n\nIt is important to keep in mind that this classification is based on probabilities and similarity to historical articles, and does no analysis on author intent. An ML-classifier should never be used as the sole source of decisions.")
            else:
                st.write("This article has been flagged as **Misinformation**. \n\nThis means that the classifier found high similarity between the article and other articles known to be misinformation. \n\nThe difference between disinformation and misinformation has to do with author intent. The classification of this article indicate **low** probability of author malintent.\n\nIt is important to keep in mind that this classification is based on probabilities and similarity to historical articles, and does no analysis on actual author intent. An ML-classifier should never be used as the sole source of decisions.")

##########################################################################################################################################
##########################################################################################################################################
if page == "Legal Helper":
# Set up OpenAI API key
# REDACTED

    st.title("Legal Framework Resource")
    st.write("This tool will help you better understand the EU legalization frameworks. Simply enter your question about the documents in the box. Based on our predefined prompts sent to Chat-GPT, it will scan the legal document(s) and provide an answer to your question.")


    left_column, right_column = st.columns([1,2])

    # Sample dictionary containing framework to website mapping
    framework_websites = {
        "GDPR": "https://gdpr-info.eu/",
        "AI Act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A52021PC0206",
        "NIS 2 Directive": "https://eur-lex.europa.eu/eli/dir/2022/2555",
        "2019 Cybersecurity Act": "https://eur-lex.europa.eu/eli/reg/2019/881/oj",
        "e-Evidence Act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R1543", #osäker på denna länken
        "Digital Service Act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32022R2065",
        "AI Liability Act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52022PC0496",
        # ... add other documents and their corresponding websites here ...
    }

    def get_framework_description(framework):
        descriptions = {
            "GDPR": "The General Data Protection Regulation (GDPR) is a regulation in EU law on data protection and privacy in the European Union and the European Economic Area.",
            "AI Act": "The AI Act proposes regulations for artificial intelligence in the European Union.",
            "NIS 2 Directive": "The Directive on Security of Network and Information Systems (NIS 2) is the first piece of EU-wide legislation on cybersecurity.",
            "2019 Cybersecurity Act": "The Cybersecurity Act reinforces the mandate of the European Union Agency for Cybersecurity (ENISA).",
            "e-Evidence Act": "The e-Evidence Act pertains to electronic evidence in criminal matters.",
            "Digital Service Act": "The Digital Service Act aims to create a safer digital space.",
            "AI Liability Act": "The AI Liability Act is a conceptual framework for AI operations.",
            "Blank": "With Blank, the search will be general using all documents."
        }
        website_url = framework_websites.get(framework, None)
        return descriptions.get(framework, "Description not available."), website_url

    with left_column:
        #st.sidebar.title("Navigation")
        #st.sidebar.write("""
        #- [Disinformation Checker](#)
        #- [Legal Surfer](#)
        #- [Metrics & Graphs](#)
        #- [About the App](#)
        #""")
        st.info("\n\nSelect a specific legal framework to research below, or leave on 'Blank' to post  general question related to all relevant frameworks. \n\nThen select your level of expertise (Advanced - You have knowledge of key concepts and the relevant legal frameworks. Simplified - you want extra support in the answer). \n\nAfter this, send your question by pressing the button below.")
        
        legal_frameworks = ["Blank", "GDPR", "AI Act", "NIS 2 Directive",
                            "2019 Cybersecurity Act", "e-Evidence Act", "Digital Service Act", "AI Liability Act"]
        
        selected_framework = st.selectbox("Choose a Legal Framework:", legal_frameworks)
        description, website_url = get_framework_description(selected_framework)
        st.write(description)
        
        if website_url:
            st.write(f"[Source for {selected_framework}]({website_url})")

    with right_column:
        #st.write("Choose Your Expertise Level")
        expertise_level = st.radio("Select your level of expertise/the level of depth you want in the answer:", ("Advanced", "Simplified"))

        if expertise_level == "Advanced":
            st.write("You selected 'Advanced'. Enter your legal question in free format:")
            legal_question = st.text_area("Ask question here:")
        else:
            st.write("You selected 'Simplified'. Enter your legal question in free format:")
            legal_question = st.text_area("Ask question here:")

        # Add a "Send Question" button
        if st.button("Send Question"):
            if legal_question:
                # Create system messages and the user's message for the ChatGPT API
                messages = [
                    {"role": "system", "content": "You are helping EU law makers and decision makers with interpreting legal documents, especially regarding questions of disinformation, fake news and malinformation."},
                    {"role": "system", "content": "Keep the answers concise and factual, but assume that the reader is at expert level and have deep knowledge about key concepts and the relevant legal frameworks."},
                    {"role": "system", "content": "base your answers on the following legal frameworks: GDPR, AI Act, NIS 2 Directive, 2019 Cybersecurity Act, e-Evidence Act, Digital Service Act, AI Liability Act"},
                    {"role": "system", "content": "at the end of the reply, always note any sources you refer to or use in the response. when possible, mention the website to the sources."},
                ]

                if expertise_level == "Simplified":
                    # Append the message for Novice users
                    messages.append({"role": "user", "content": "Use a simplified language in all your answers, assume that the user do not have knowledge about key concepts and need some extra support in your answer."})


                # Add an additional system message if a specific framework is selected
                if selected_framework != "Blank":
                    messages.append({"role": "system", "content": f"Base your answer mainly on this legal framework {selected_framework}, how does the question relate to this framework?"})
                
                messages.append({"role": "user", "content": legal_question})

                # Use the ChatGPT API with the messages
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.3
                )
                
                response = completion.choices[0].message.content
                right_column.write(response)
##########################################################################################################################################
##########################################################################################################################################
if page == "About":
    st.markdown("# About")
    st.subheader("The Project")
    st.write("""This Streamlit page is created for a project which is a part of the course "Introduction to Human-centered AI” at the University of Gothenburg. We've had the privilege to collaborate with the Research Institutes of Sweden (RISE) for this project. Our wish is to contribute to the recently initiated research initiative, ATHENA, carried out by RISE in collaboration with the European Union's institutions and research partners.""")

    st.subheader("ATHENA Project Details")
    st.write("""ATHENA stands for "An Exposition on Foreign Information Manipulation and Interference." It's a project by the European Union (EU) that includes 14 organizations from 11 countries. The research company Trilateral leads the project. Its main goal is to protect democratic processes in the European Union (EU) from foreign information manipulation and interference (FIMI).
    """)
    st.write("""The project uses machine learning algorithms, conducts field studies, and develops detection tools to help the EU find, study, and fight against FIMI. By creating these tools and understanding the impact of FIMI on society, ATHENA helps policymakers, businesses, and community groups take action against FIMI.""")

    st.subheader("The Machine Learning Model")
    st.write("We experimented with three different models before making a final decision on which one to choose. Ultimately, we selected the [Logistic Regression model](https://www.ibm.com/topics/logistic-regression), while also testing the k-NN (k-Nearest Neighbors) and Naive Bayes models. We evaluated each model both with and without applying the Bag of Words (BoW) technique. Additionally, we performed cross-validation on all models to ensure we selected the one with the highest performance score.")

    st.write("**Article credibility analysis**")
    st.write("The ML classifier is based on logistic regression due to when cross-valuated this type of statistical model had the best accuracy scores for our type of dataset. The model is trained on the WELfake dataset (more information on WELfake [here](https://ieeexplore.ieee.org/document/9395133)), a large dataset of over 72 000 articles, half of the data is non-credible news and the other half is verified news. Our model is trained on 80% of the dataset and tested on the remaining 20%. When the model is trained the model gets the label (true or false) as well as the text of the articles. The model then predicts connections between these two and tests the predictions on the last 20% without adjusting, the model correctly assesses articles as non-credible or credible 96% of the time with a test of over 14 000 articles.")
    
    st.write("**Disinformation classifier**")
    st.write("Experimental ML tool. The aim of this tool was to explore the possibilities of using machine learning to distinguish between misinformation and disinformation. The experiment provided a more nuanced understanding of the challenges of detecting intent in disinformation and the amount of work required for unbiased data. The dataset used is a combination of the WELfake-dataset and a scraped version of the EUvsDisinfo-dataset (more information on WELfake [here] (https://ieeexplore.ieee.org/document/9395133) and EuvsDisinfo [here](https://euvsdisinfo.eu/disinformation-cases/)). The WELfake dataset has been reduced to only the non-credible articles and the number of articles has been balanced in to match the disinformation dataset of 15 000 articles. A total of 30 000 articles were used where 80% is a training set and 20% is the testing set. Reaching a testing accuracy of 95%. The problem with this dataset is that we know that at least 50% of the articles are correctly labeled as disinformation. The remaining 50% is labeled as “fake news” or “non-credible” by the creator of the WELfake dataset and could thereby be a mix of misinformation and disinformation making it unclear what is being predicted. See this model as a “proof of concept” or an early prototype with a need for a more robust dataset.")
    st.write("The dataset used is a combination of the WELfake-dataset and a scraped version of the EUvsDisinfo-dataset (more information on WELfake [here](https://ieeexplore.ieee.org/document/9395133) and EuvsDisinfo [here](https://euvsdisinfo.eu/disinformation-cases/)). The WELfake dataset has been reduced to only the non-credible articles and the number of articles has been balanced in to match the disinformation dataset of 15 000 articles. A total of 30 000 articles were used where 80% is a training set and 20% is the testing set. Reaching a testing accuracy of 95%. The problem with this dataset is that we know that at least 50% of the articles are correctly labeled as disinformation. The remaining 50% is labeled as “fake news” or “non-credible” by the creator of the WELfake dataset and could thereby be a mix of misinformation and disinformation making it unclear what is being predicted. See this model as a “proof of concept” or an early prototype with a need for a more robust dataset.")
    
    st.subheader("**Transparency Note**")
    st.write("As students of the University of Gothenburg's masters program “Human-centered AI” a central part of our project is a steadfast commitment to transparency. It's crucial for users to understand how our model works, its strengths, and its limitations. By being transparent about these details we can ensure that users have the necessary context to interpret the model's outputs. Below you can read further into our metrics on our “article credibility analysis model:")
    
    st.write("**Learning Accuracy:** Our model correctly learned from the examples 99.9% of the time.")
    st.write("**Testing Accuracy:** On new articles, it correctly identified \"fake\" news 96.12% of the time.")
    st.write("**Mistakes:** It wrongly called a real article \"fake\" 5.64% of the time. It wrongly called a fake article \"real\" 2.15% of the time.") # Used FPR and FNR here
    st.write("**Fairness:** Our model was consistent in its decisions, correctly identifying fake news 97.85% of the time across different articles.") # Used EO here
    st.write("**Trustworthiness:** If the model says an article is \"fake,\" it's right 94.89% of the time. If the model says an article is \"real,\" it's right 97.62% of the time.") # Used PPV and NPV here
    st.write("""Our news-checking model is reliable and fair in its predictions, but it's always wise to double-check news from other sources.
    """)
    
    st.subheader("Team Members")
    st.write("- Jonathan Häggqvist: gushagjoah@student.gu.se")
    st.write("- Magnus Wahlström: gusmagnwah@student.gu.se")
    st.write("- Linus Zetterlund: guslinuze@student.gu.se")
    st.write("- Ebba Rydnell: gusrydeb@student.gu.se")
    st.write("For more about our MSc program, [you can check out the page for the program](https://www.gu.se/en/study-gothenburg/human-centered-artificial-intelligence-masters-programme-t2hai).")
