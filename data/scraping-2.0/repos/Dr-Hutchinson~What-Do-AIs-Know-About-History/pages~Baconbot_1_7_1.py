import os
import openai
import streamlit as st
from datetime import datetime as dt
import pandas as pd
from numpy import mean
import streamlit_authenticator as stauth
import pygsheets
from google.oauth2 import service_account
import ssl


#st.set_page_config(
    #page_title='Simulated Conversations with Francis Bacon',
    #layout='wide',
    #page_icon='🔍'
#)

def app():

    st.title("BaconBot: An AI Imitation of Francis Bacon")
    st.header("Open Interface")
    st.subheader("Access to the Open Interface for BaconBot is currently limited. Click here to apply for access.")

    st.write("The following version of BaconBot permits users to compose their own questions and customize the question parameters. The login panel will load in a moment.")
    #pygsheets credentials for Google Sheets API
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

    gc = pygsheets.authorize(custom_credentials=credentials)

    #login setup for streamlit_authenticator via Google Sheets API
    sh0 = gc.open('users')
    wks0 = sh0[0]
    database_length = wks0.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
    end_row0 = str(len(database_length))
    usernames_grab = wks0.get_as_df(has_header=False, index_column=0, start='A2', end=('A'+end_row0), numerize=False)
    usernames_list = usernames_grab.values.tolist()
    access_grab= wks0.get_as_df(has_header=False, index_column=0, start='B2', end=('B'+end_row0), numerize=False)
    access_list = access_grab.values.tolist()
    names_grab = wks0.get_as_df(has_header=False, index_column=0, start='C2', end=('C'+end_row0), numerize=False)
    names_list = names_grab.values.tolist()
    user_id_grab = wks0.get_as_df(has_header=False, index_column=0, start='D2', end=('D'+end_row0), numerize=False)
    user_id_list = names_grab.values.tolist()

    #streamlit_authenticator login
    names =  [lst[0] for lst in names_list]
    usernames = [lst[0] for lst in usernames_list]
    passwords = [lst[0] for lst in access_list]
    user_ids = [lst[0] for lst in user_id_list]

    hashed_passwords = stauth.Hasher(passwords).generate()


    authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
        'some_cookie_name', 'some_signature_key', cookie_expiry_days=300)

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write('Welcome *%s*' % (name))

        with st.sidebar:
            st.write('Explore more about the life and times of Francis Bacon:')
            st.write('[Six Degrees of Francis Bacon](http://www.sixdegreesoffrancisbacon.com/), Carnegie Mellon University')
            st.write('[Jürgen Klein and Guido Giglioni, "Francis Bacon", The Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/francis-bacon/)')
            st.write('[Richard S. Westfall, "Francis Bacon", The Galileo Project, Rice University](http://galileo.rice.edu/Catalog/NewFiles/bacon.html)')

        #Begin Baconbot code
        st.title('Simulated Conversations with Francis Bacon')
        col1, col2 = st.columns([3.0,3.5])
        with col2:
            bacon_pic = st.image(image='./bacon.png', caption="Portrait of Francis Bacon. National Portrait Gallery, London.")



        def button_one():
            st.write("Choose the settings below, and then pose questions to Francis Bacon. See this link for more information on these settings.")

            with st.form('form2'):

                prompt_choice_freeform = "I am a representation of Francis Bacon, a key figure in the Scientific Revolution. You can ask me questions and I will answer in the style of Bacon's Novum Organum."
                prompt_choice_rationale = "I am an AI representation of Francis Bacon, a key figure in the early modern period. I will reply to your questions, and provide a historical rationale for my response."
                prompt_choice_haiku = "I am Lord Francis Bacon, a key figure in reign of King James I of England. I will answer your questions in the form of a haiku in a 5-7-5 syllabic structure."

                model_choice = st.radio("Select AI model. GPT-3 is the general purpose AI model. The Novum Organum model is a GPT-3 fine-tuned on Bacon's classic work of scientific theory.", ["GPT-3: Davinci Engine model", "Novum Organum model"])
                prompt_choice = st.radio('Select Prompt.', [prompt_choice_freeform, prompt_choice_rationale, prompt_choice_haiku])

                with st.expander("Advanced Settings:"):
                    prompt_booster = st.radio("Zero Shot vs. Few Shot Prompting. If you chose one of the prompt boosters below, the AI model will be given pre-selected examples of the type of prompt you want to submit, increasing the chance of a better reply. However, this will also increase the chance the reply will repeat the booster choice. Choose 'None' to field questions without a booster.", ["None", "Question Booster", "Rationale Booster", "Haiku Booster"])

                temperature_dial = st.slider("Temperature Dial. Lower values are generally more accurate, but lower extremes yield more repetitive replies. Higher values are more creative, but higher extremes result in incoherent responses.", 0.0, 1.0)
                response_length = st.slider("Response Length. Recommended range is 75-150 for general questions, 150-250 for rationale questions, and 25-50 for haikus.", 25, 250)
                submission_text = st.text_area("Enter your questions and comments below to Francis Bacon in this space. Be patient as he considers your statement.", max_chars=1000)
                submit_button_1 = st.form_submit_button(label='Submit Question')

                if submit_button_1:
                    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
                    now = dt.now()

                    #match user with user_id for Openai submission
                    @st.cache(ttl=6000)
                    def user_id_lookup():
                        sh_id = gc.open('users')
                        wks_id = sh_id[0]
                        database_length = wks_id.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                        end_row0 = str(len(database_length))
                        df = wks_id.get_as_df(has_header=True, index_column=None, start='A1', end=('D'+end_row0), numerize=False)
                        user_info = df.loc[df['names'] == name]
                        index_info = df.index.values[df['names']==name]
                        index_str = ' '.join(str(x) for x in index_info)
                        index_number = int(float(index_str))
                        user_id = user_info.at[index_number, 'user_id']
                        return user_id

                    user_id = user_id_lookup()

                    #model selection for OpenAI query
                    if model_choice == "GPT-3: Davinci Engine":
                        model_select = 'text-davinci-003'
                    else:
                        model_select = st.secrets['novum_organum_model']

                    prompt_boost_haiku_1 = "Compose a haiku on the events in London during the spring of 1610."
                    prompt_boost_haiku_2 = "Haiku: The taverns are full of talk, Of the moons of Jupiter and of the Prince’s ship."
                    prompt_boost_haiku_3 = "Compose a haiku in the style of Basho."
                    prompt_boost_haiku_4 = "Haiku: On a withered branch, A crow has alighted, Nightfall in autumn."
                    prompt_boost_rationale_1 = "Question: Could you describe your impression of the scientific investigation of figures from antiquity like Galen?"
                    prompt_boost_rationale_2 = "Answer: Galen was a great man, but he had not the advantage of a good method. His idols of the market place, as I have called them, were his errors and fancies, which have misled some and perverted others. He trusted too much to authority and to received systems, and too little to the examination of particulars. He was a practitioner and not a philosopher, and was therefore more intent upon practice than upon theory; and yet he was not a mere empiric. He was a great observer, and a man of clear sense and great experience, but he was a slave to the logic and philosophy of his age, and therefore was very deficient in the true principles of induction."
                    prompt_boost_rationale_3 = "Rationale: The critique of an ancient authority in medicine on the basis of his inadequate method is keeping with an important theme in Novum Organum and Bacon’s larger scientific philosophy. The specific mention of the “Idols of the Marketplace” is an important concept in the Novum Organum."
                    prompt_boost_rationale_4 = "Question: What do you see as the hallmarks of the New Science?"
                    prompt_boost_rationale_5 = "Answer: The New Science (as I would like to call it, so as not to offend the old) has two main indications. The first is to discover the occasions and causes of nature’s productions and actions; the second, by careful and well-ordered experiments (such as are derived from the light of nature), to acquire a competent knowledge of the power and office of each production and action."
                    prompt_boost_rationale_6 = "Rationale: The generated response outlines one of the major contributions of Francis Bacon to the philosophy of science, what would become the modern scientific method."
                    prompt_boost_question_1 = "Question: What do you see as the hallmarks of the New Science?"
                    prompt_boost_question_2 = "Answer: The New Science (as I would like to call it, so as not to offend the old) has two main indications. The first is to discover the occasions and causes of nature’s productions and actions; the second, by careful and well-ordered experiments (such as are derived from the light of nature), to acquire a competent knowledge of the power and office of each production and action."


                    if prompt_booster == "None":
                        prompt_text = prompt_choice + "\n\nQ:"
                    elif prompt_booster == "Rationale Booster":
                        prompt_text = prompt_choice + "\n\n" + prompt_boost_rationale_1 + "\n\n" + prompt_boost_rationale_2 + "\n\n" + prompt_boost_rationale_3 + "\n\n" + prompt_boost_rationale_4 + "\n\n" + prompt_boost_rationale_5 + "\n\n" + prompt_boost_rationale_6 + "\n\n" + "Question:"
                    elif prompt_booster == "Haiku Booster":
                        prompt_text = prompt_choice + "\n\n" + prompt_boost_haiku_1 + "\n\n" + prompt_boost_haiku_2 + "\n\n" + prompt_boost_haiku_3 + "\n\n" + prompt_boost_haiku_4
                    else:
                        prompt_text = prompt_choice + "\n\n" + prompt_boost_question_1 + "\n\n" + prompt_boost_question_2 + "\n\n" + "Question:"

                    #prompt_text = prompt_choice + "\n\nQ:"

                    openai.api_key = os.getenv("OPENAI_API_KEY")

                    summon = openai.Completion.create(
                        model=model_select,
                        prompt= prompt_text + " " + submission_text,
                        temperature=temperature_dial,
                        user=user_id,
                        max_tokens=response_length)

                    response_json = len(summon["choices"])

                    for item in range(response_json):
                        output = summon['choices'][item]['text']

                    response = openai.Completion.create(
                            engine="content-filter-alpha",
                            prompt= "<|endoftext|>"+output+"\n--\nLabel:",
                            temperature=0,
                            max_tokens=1,
                            user=user_id,
                            top_p=0,
                            logprobs=10)

                    output_label = response["choices"][0]["text"]

                # OpenAI Content Filter code - comments in this section from OpenAI documentation: https://beta.openai.com/docs/engines/content-filter
                    # This is the probability at which we evaluate that a "2" is likely real
                        # vs. should be discarded as a false positive

                    def filter_function():
                        output_label = response["choices"][0]["text"]
                        toxic_threshold = -0.355

                        if output_label == "2":
                            # If the model returns "2", return its confidence in 2 or other output-labels
                            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

                            # If the model is not sufficiently confident in "2",
                            # choose the most probable of "0" or "1"
                            # Guaranteed to have a confidence for 2 since this was the selected token.
                            if logprobs["2"] < toxic_threshold:
                                logprob_0 = logprobs.get("0", None)
                                logprob_1 = logprobs.get("1", None)

                                # If both "0" and "1" have probabilities, set the output label
                                # to whichever is most probable
                                if logprob_0 is not None and logprob_1 is not None:
                                    if logprob_0 >= logprob_1:
                                        output_label = "0"
                                    else:
                                        output_label = "1"
                                # If only one of them is found, set output label to that one
                                elif logprob_0 is not None:
                                    output_label = "0"
                                elif logprob_1 is not None:
                                    output_label = "1"

                                # If neither "0" or "1" are available, stick with "2"
                                # by leaving output_label unchanged.

                        # if the most probable token is none of "0", "1", or "2"
                        # this should be set as unsafe
                        if output_label not in ["0", "1", "2"]:
                            output_label = "2"

                        return output_label

                    # filter or display OpenAI outputs, record outputs to Google Sheets API
                    if int(filter_function()) < 2:
                        st.write("Bacon's Response:")
                        st.write(output)
                        st.write("\n\n\n\n")
                        st.subheader('As Lord Bacon says, "Truth will sooner come out from error than from confusion."  Please click on the Rank Bacon button above to rank this reply for future improvement.')
                    elif int(filter_function()) == 2:
                        st.write("The OpenAI content filter ranks Bacon's response as potentially offensive. Per OpenAI's use policies, potentially offensive responses will not be displayed. Consider adjusting the question or temperature, and ask again.")

                    st.write("\n\n\n\n")
                    st.write("OpenAI's Content Filter Ranking: " +  output_label)

                    def total_output_collection():
                        d1 = {'user':[name], 'user_id':[user_id], 'model':[model_choice], 'prompt':[prompt_choice], 'prompt_boost':[prompt_booster], 'question':[submission_text], 'output':[output], 'temperature':[temperature_dial], 'response_length':[response_length], 'filter_ranking':[output_label], 'date':[now]}
                        df1 = pd.DataFrame(data=d1, index=None)
                        sh1 = gc.open('bacon_outputs')
                        wks1 = sh1[0]
                        cells1 = wks1.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                        end_row1 = len(cells1)
                        wks1.set_dataframe(df1,(end_row1+1,1), copy_head=False, extend=True)

                    def output_collection_filtered():
                        d2 = {'user':[name], 'user_id':[user_id], 'model':[model_choice], 'prompt':[prompt_choice], 'prompt_boost':[prompt_booster], 'question':[submission_text], 'output':[output], 'temperature':[temperature_dial], 'response_length':[response_length], 'filter_ranking':[output_label], 'date':[now]}
                        df2 = pd.DataFrame(data=d2, index=None)
                        sh2 = gc.open('bacon_outputs_filtered')
                        wks2 = sh2[0]
                        cells2 = wks2.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                        end_row2 = len(cells2)
                        wks2.set_dataframe(df2,(end_row2+1,1), copy_head=False, extend=True)

                    def temp_output_collection():
                        d3 = {'user':[name], 'user_id':[user_id], 'model':[model_choice], 'prompt':[prompt_choice], 'prompt_boost':[prompt_booster], 'question':[submission_text], 'output':[output], 'temperature':[temperature_dial], 'response_length':[response_length], 'filter_ranking':[output_label], 'date':[now]}
                        df3 = pd.DataFrame(data=d3, index=None)
                        sh3 = gc.open('bacon_outputs_temp')
                        wks3 = sh3[0]
                        wks3.set_dataframe(df3,(1,1))

                    if int(filter_function()) == 2:
                        output_collection_filtered()
                        total_output_collection()
                    else:
                        temp_output_collection()
                        total_output_collection()

        def button_two():
            #Rank Bacon_bot Responses
            st.write("Rank Bacon's Reply:")
            sh1 = gc.open('bacon_outputs_temp')

            wks1 = sh1[0]
            submission_text = wks1.get_value('F2')
            output = wks1.get_value('G2')
            prompt_text = wks1.get_value('D2')
            st.subheader('Prompt:')
            st.write(prompt_text)
            st.subheader('Your Question')
            st.write(submission_text)
            st.subheader("Bacon's Reply:")
            st.write(output)


            with st.form('form2'):
                bacon_score = st.slider("How much does the reply resemble the style of Francis Bacon?", 0, 10, key='bacon')
                worldview_score = st.slider("Is the reply consistent with Bacon's worldview?", 0, 10, key='worldview')
                accuracy_rank = st.slider("Does the reply appear factually accurate?", 0, 10, key='accuracy')
                coherence_rank = st.slider("How coherent and well-written is the reply?", 0,10, key='coherence')
                st.write("Transmitting the rankings takes a few moments. Thank you for your patience.")
                submit_button_2 = st.form_submit_button(label='Submit Ranking')

                if submit_button_2:
                    sh1 = gc.open('bacon_outputs_temp')
                    wks1 = sh1[0]
                    df = wks1.get_as_df(has_header=True, index_column=None, start='A1', end=('K2'), numerize=False)
                    name = df['user'][0]
                    user_id = df['user_id'][0]
                    model_choice = df['model'][0]
                    prompt_choice = df['prompt'][0]
                    prompt_boost = df['prompt_boost'][0]
                    submission_text = df['question'][0]
                    output = df['output'][0]
                    temperature_dial = df['temperature'][0]
                    response_length = df['response_length'][0]
                    output_label = df['filter_ranking'][0]
                    now = dt.now()
                    ranking_score = [bacon_score, worldview_score, accuracy_rank, coherence_rank]
                    ranking_average = mean(ranking_score)

                    def ranking_collection():
                        d4 = {'user':[name], 'user_id':[user_id],'model':[model_choice], 'prompt':[prompt_choice], 'prompt_boost':[prompt_boost],'question':[submission_text], 'output':[output], 'temperature':[temperature_dial], 'response_length':[response_length], 'filter_ranking':[output_label], 'bacon_score':[bacon_score], 'worldview_score':[worldview_score],'accuracy_rank':[accuracy_rank], 'coherence':[coherence_rank], 'overall_ranking':[ranking_average], 'date':[now]}
                        df4 = pd.DataFrame(data=d4, index=None)
                        sh4 = gc.open('bacon_rankings')
                        wks4 = sh4[0]
                        cells4 = wks4.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                        end_row4 = len(cells4)
                        wks4.set_dataframe(df4,(end_row4+1,1), copy_head=False, extend=True)

                    ranking_collection()
                    st.write('Rankings recorded - thank you! Feel free to continue your conversation with Francis Bacon.')

        with col1:

            st.write("Select the 'Ask Bacon' button to ask the AI questions. Select 'Rank Bacon' to note your impressions of his responses.")

            pages = {
                0 : button_one,
                1 : button_two,
            }

            if "current" not in st.session_state:

                st.session_state.current = None

            if st.button("Ask Bacon"):
                st.session_state.current = 0
            if st.button("Rank Bacon"):
                st.session_state.current = 1

            if st.session_state.current != None:
                pages[st.session_state.current]()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
