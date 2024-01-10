# All imports

import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Iterative Optimizer", page_icon = "🧠", layout = "wide")
st.header("🧠 Iterative Optimizer")

add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

import openai
import pandas as pd

## Importing functions
from core.part2_asset_score_calculator import get_unique_kristal_name, read_portfolios, show_particular_portfolio, returns_preprocessing, reading_returns_file, read_portfolios, read_kristal_mappings, download_data_as_excel_link, get_unique_client_ids, create_prices_dataframe, create_returns_dataframe, create_stats_dataframe, portfolio_manipulation, write_results_to_excel_file, obtain_respective_asset_value, read_portfolios_for_particular_client, kristal_mapping_manipulation

### CODE

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY

# Error handling for OpenAI API key
if not openai_api_key:
    st.warning(
        "There is something wrong with the API Key Configuration."
        "Please check with creator of the program (OpenAI keys can be found at https://platform.openai.com/account/api-keys)"
    )

if 'username' in st.session_state:
    st.session_state.username = st.session_state.username

# st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# Display app only if user is logged in
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Call the get_unique_client_ids function
    portfolios = read_portfolios()
    client_ids = get_unique_client_ids(portfolios)

    kristal_mappings = read_kristal_mappings()
    kristal_name = get_unique_kristal_name(kristal_mappings)

    # Allow user to select client id
    client_id_selected = st.selectbox(label = "Please select the client id from dropdown", options = client_ids, index = None, key = "select_client_id", help = "select_client_id", placeholder="Choose particular client id", disabled = False, label_visibility = "visible")

    if client_id_selected:

        # Select filtered_df
        filtered_df = read_portfolios_for_particular_client(portfolios, client_id_selected)
        # filtered_df = show_particular_portfolio(portfolios, target_client_id = client_id_selected)

        # Display dataframe containing final results
        st.dataframe(data = filtered_df, use_container_width = True, column_order = None)
    

    # Allow user to select client id
    kristal_name_selected = st.selectbox(label = "Please select the kristal name from dropdown", options = kristal_name, index = None, key = "select_kristal_name", help = "Please choose the kristal name for which you would like to analyze", placeholder="Choose particular kristal name", disabled = False, label_visibility = "visible")

    if kristal_name_selected:
        asset_ticker = obtain_respective_asset_value(kristal_mappings, kristal_name_selected)
        st.write("Asset selected:", asset_ticker)

    # Allow user to select date range
    start_date = pd.to_datetime('2019-03-29')
    end_date = pd.to_datetime('2023-08-31')

    # date_selected = st.date_input(
    #     label = "Please select a date range for which you want to analyze portfolio data for",
    #     value = (start_date, end_date),
    #     min_value = None,
    #     max_value = None,
    #     key = "select_date_input",
    #     help = "Please make sure to select a date range/interval",
    #     on_change = None,
    #     disabled = False,
    #     format = "DD/MM/YYYY",
    #     label_visibility = "visible"
    #     )

    # if not isinstance(date_selected, tuple):
    #     print("Error: date_selected is not a tuple.")
    #     st.warning('date_selected is not a tuple', icon="⚠️")
    #     st.stop()

    # else:
    #     if len(date_selected) != 2:

    #         if len(date_selected) == 0:
    #             print("Error: date_selected is an empty tuple.")
    #             st.warning('Please make sure to select a date range (and not leave it empty)', icon="⚠️")
    #             st.stop()

    #         else:
    #             print("Error: date_selected is not a tuple of length 2.")
    #             st.warning('Please make sure to select an interval of dates (that is, both start date and end date)', icon="⚠️")
    #             st.stop()

    #     # Length of tuple is equal to 2
    #     else:

    #         # Convert the tuple of dates to the desired format
    #         selected_start_date = pd.to_datetime(date_selected[0]).strftime('%Y-%m-%d')
    #         selected_end_date = pd.to_datetime(date_selected[1]).strftime('%Y-%m-%d')

    # If user clicks on the button process
    if st.button("Run", type = "primary"):

        if not client_id_selected:
            st.warning("Please select client id", icon = "⚠️")

        if not kristal_name_selected:
            st.warning("Please select kristal name", icon = "⚠️")
        
        # if not date_selected:
        #     st.warning("Please select date range", icon = "⚠️")

        if client_id_selected and kristal_name_selected:

            # with st.spinner("Generating Recommendations"):
            #     returns = reading_returns()
            #     returns = returns_manipulation(returns, start_date, end_date)
            #     portfolio = read_portfolios()
            #     kristal_mappings, asset_universe, adjusted_returns_universe = read_kristal_mappings()
            #     excel_file_list = []
            #     num_clients = len(client_ids)
            #     for i in range(num_clients):
            #         gen_clients_ranking(client_ids[i])

            with st.spinner("Reading asset returns excel file"):
                returns = reading_returns_file()
                # returns = returns_preprocessing(returns, start_date, end_date)
                returns = returns_preprocessing(returns, start_date, end_date)

            st.success("Successfully read asset returns excel file", icon="✅")

            with st.spinner("Reading Asset Index mapping excel file"):
                kristal_mappings, asset_universe, adjusted_returns_universe = kristal_mapping_manipulation(kristal_mappings, returns)

            st.success("Successfully read Asset Index mapping excel file", icon="✅")

            with st.spinner("Reading portfolios excel file"):
                portfolio1 = read_portfolios_for_particular_client(portfolios = portfolios, client_id = client_id_selected)

            st.success("Successfully read portfolios excel file", icon="✅")

            with st.spinner("Analyzing Old & New Portfolios:"):
                dates, port_prices, new_port_prices, new_asset_weights, cagr, new_cagr, len_prices, len_new_port_prices, port_returns, new_port_returns, end_price, new_end_price = portfolio_manipulation(portfolio1, returns, kristal_mappings, asset_universe, asset_ticker)
                prices_df = create_prices_dataframe(dates, port_prices, asset_ticker, new_port_prices, new_asset_weights, returns)
                returns_df = create_returns_dataframe(returns, port_returns, new_port_returns, asset_ticker)
                stats_df = create_stats_dataframe(cagr, new_cagr, end_price, new_end_price, port_returns, len_prices, len_new_port_prices, returns_df, prices_df, new_port_returns, port_prices, new_port_prices, dates, returns)

                # excel_file_list = []
                filepath = write_results_to_excel_file(prices_df, returns_df, stats_df, client_id_selected, asset_ticker)

            st.success("Successfully analyzed old & new portfolios", icon="✅")

        # Display "Scores" excel sheet from results excel file
        st.dataframe(data = stats_df, use_container_width = True, column_order = None)

        # Download excel data
        download_data_as_excel_link(filepath)

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")