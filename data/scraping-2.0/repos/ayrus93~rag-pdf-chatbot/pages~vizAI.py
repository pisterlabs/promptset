import pandas as pd
import openai
import streamlit as st
import warnings


def run_request(question_to_ask, model_type):
    if model_type == "gpt-4":
        task = "Generate Python Code Script. The script should only include code, no comments."
    elif model_type == "gpt-3.5-turbo":
        task = "Generate Python Code Script."
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo":
        # Run ChatGPT API
        openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {"role":"system","content":task},
                {"role":"user","content":question_to_ask}])
        res = response["choices"][0]["message"]["content"]

    # rejig the response

    res = format_response(res)
    return res



def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question):
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code

def main():
    warnings.filterwarnings("ignore")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(layout="wide",page_title="vizAI")
    st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
                VizAI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations using Natural Language \
                with ChatGPT</h2>", unsafe_allow_html=True)

    # List to hold datasets
    if "GenSQL_op_df" not in st.session_state:
        st.error('Run query in GenSQL to visualize')
    else:
        # use the list already loaded
        datasets = st.session_state["GenSQL_op_df"]
        st.dataframe(datasets)

    # Text area for query
    question = st.text_area(":eyes: What would you like to visualise?",height=10)
    go_btn = st.button("Go...")

    # Make a list of the models which have been selected
    #model_dict = {model_name: use_model for model_name, use_model in use_model.items() if use_model}
    #model_count = len(model_dict)

    # Execute chatbot query
    if go_btn:
        # Place for plots depending on how many models
        # Get the primer for this dataset
        primer1,primer2 = get_primer(datasets,'datasets')
        # Format the question
        question_to_ask = format_question(primer1,primer2 , question)    
        # Create model, run the request and print the results

        try:
            # Run the question
            answer=""
            answer = run_request(question_to_ask, "gpt-3.5-turbo")
            # the answer is the completed Python script so add to the beginning of the script to it.
            answer = primer2 + answer
            plot_area = st.empty()
            plot_area.pyplot(exec(answer))           
        except Exception as e:
            if type(e) == openai.error.APIError:
                st.error("OpenAI API Error. Please try again a short time later.")
            elif type(e) == openai.error.Timeout:
                st.error("OpenAI API Error. Your request timed out. Please try again a short time later.")
            elif type(e) == openai.error.RateLimitError:
                st.error("OpenAI API Error. You have exceeded your assigned rate limit.")
            elif type(e) == openai.error.APIConnectionError:
                st.error("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings.")
            elif type(e) == openai.error.InvalidRequestError:
                st.error("OpenAI API Error. Your request was malformed or missing required parameters.")
            elif type(e) == openai.error.AuthenticationError:
                st.error("Please enter a valid OpenAI API Key.")
            elif type(e) == openai.error.ServiceUnavailableError:
                st.error("OpenAI Service is currently unavailable. Please try again a short time later.")                   
            else:
                st.error("Unfortunately the code generated from the model contained errors and was unable to execute. ")

    # Display the datasets in a list of tabs
    # Create the tabs




    # Hide menu and footer
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()