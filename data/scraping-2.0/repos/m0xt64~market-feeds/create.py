import os
import json
import datetime
import streamlit as st
import openai as openai
from openai import OpenAI

# Function to set up the sidebar
def setup_sidebar():
    st.sidebar.markdown('# Market Feeds Creator')
    st.sidebar.markdown("##### Create and Contribute to the Market Feeds.")
    st.sidebar.markdown("###### See help icon for tips and templates.")

    user_info = {
        'created_by': st.sidebar.text_input("Your nickname:",help="Enter a unique nickname that will be used to identify your contributions.")  
    }

    api_info = {
        'api_key': st.sidebar.text_input("Enter your OpenAI API key:", type="password", help="Paste your OpenAI API key here. It's kept secret. We won't have access to it. It serves to authenticate your requests to OpenAI for this testing purpose only. Once your market feed is created and approved, we will use our own API key to generate the feed.")
    }

    model_info = {
        'model': st.sidebar.selectbox("Select a model", ["gpt-4"]),
        'temperature': st.sidebar.slider("Set Temperature", min_value=0.00, max_value=2.00, value=1.00, help="We generally recommend between 0.6 and 1.0. The higher the temperature, the crazier the text. 0 means no randomness. 1 means a lot of randomness. 2 means completely random."),
    }  

    feed_info = {
        'name': st.sidebar.text_input("Feed name:", help="Enter a unique name for your feed. Examples: Ethereum Overview, Ethereum Staking: Lido"),
        'frequency': st.sidebar.selectbox("Select data source", ["weekly"]),
        'data_source': st.sidebar.selectbox("Select data source", ["Dune Analytics"]),
        'query_id': st.sidebar.text_input("Insert your query ID:", help="Enter the query ID from Dune Analytics. If your feed is approved, we will use this query ID to programatically receive the data from Dune Analytics."),
        'insert_data': st.sidebar.text_area("Insert your data:", help="Enter your query data here. Use copy paste (Ctrl+C and Ctrl+V). Before feed approval, this is a simpler and more cost-effective method to gather necessary data for feed creation."),
        'role': st.sidebar.text_area("Role:", help="Ex: You're a pro in data crunching and writing content that turns heads."),
        'goal': st.sidebar.text_area("Goal:", help="Ex: Your job is to dive into [INSERT YOUR TOPIC] data and turn what you find into a killer article."),
        'audience': st.sidebar.text_area("Audience:", help="Ex: Your readers are the users and investors who want weekly updates to stay in the loop on trends and news.Your article should offer them the must-know facts about the market, starting with some background and then spotlighting key takeaways and the latest weekly news."),
        'constraints': st.sidebar.text_area("Constraints:", help="Ex: Stick to plain English. For numbers exceeding one thousand, separate the thousands with a decimal point. You are restricted from doing any further calculations! Expected format output: Headline: (replace the Headline with your headline) Article: (replace the Article with your article)"),
        'size': st.sidebar.text_area("Size:", help="Ex: Keep it to [INSET YOUR NUMBER OF WORDS] words max. (Rcommended 120-500 words)"), 
        'instructions': st.sidebar.text_area("Instructions:", help=f"""Ex: Here are the guidelines:
        Step 1: Understanding the Dataset
        The dataset represents snapshot data for Rocket Pool at a specific point in time. Check 'week' column to set a time reference for this report, so readers are informed of the week ending by this date.
        Step 2: Tell your readers the current price of RPL by checking the 'rpl_price'. Next, explain how this price has changed in the last 7 days in terms of USD ('RPL_price_change_7d') and then in terms of Ethereum (ETH) ('RPL_price_in_ETH_change_7d').
        Step 3: Next, examine the total count of current RPL holders under 'current_holders' and observe how many holders have changed their holdings in the last 7 days ('holders_7d_change'). Consider if there is any relationship between these holder trends and the price movements.
        Step 4: Now, let's delve into liquidity analysis. Find out the amount of liquidity available on decentralized exchanges (DEXes) in terms of RPL tokens under 'current_liquidity' which is in RPL tokens. Check the weekly fluctuations in liquidity ('liquidity_7d_change'). Identify the project (DEX) with the highest liquidity by referring to 'top_project_in_liquidity' and analyze its share of the total liquidity ('top_project_liquidity_share'). Additionally, you can find the exact amount of RPL in this top liquidity pool under 'top_liquidity'.
        Step 5: Now, turn your attention to volume analysis. Determine the current trading volume for RPL tokens on decentralized exchanges (DEXes) by reviewing 'volume', which is in RPL tokens. Observe the changes in trading volume over the past week ('volume_7d_change'). Identify which project (DEX) has the highest trading volume with 'top_project_in_volume' and assess its proportion of the total volume ('top_project_volume_share'). You can also find the precise amount of RPL traded in this leading project under 'top_volume'.
        Step 6: Integrate your findings into the final article, ensuring that each paragraph encompasses a mix of different metric assessments rather than focusing on just one type per paragraph."""),
        'closing': st.sidebar.text_area("Closing:", help=f"""Ex: Make sure the content is interesting and grabs attention, but also maintain a professional tone because we're discussing financial topics.
        It would be great if readers would come back every week to read your updates. Don't forget to add some figures into the headline!""")

    }

    return user_info, api_info, model_info, feed_info

# Function to execute the model
def execute_model(api_info, model_info, feed_info):
    os.environ["OPENAI_API_KEY"] = api_info['api_key']
    system_message = " ".join([feed_info['role'], feed_info['goal'],feed_info['audience'],feed_info['constraints'],feed_info['size'],feed_info['instructions'],feed_info['closing']])
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_info['model'], 
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": feed_info['insert_data']}
            ],
        max_tokens=4000, 
        temperature=model_info['temperature'])

    return response.choices[0].message.content.strip()

def export_inputs_to_json(user_info, model_info, feed_info):
    # Use the 'name' from feed_info as the file name, adding '.json' extension
    file_name = f"{feed_info['name']}.json" if feed_info['name'] else 'unnamed_feed.json'

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Add the current date and time to the user_info dictionary
    user_info['created'] = current_datetime

    # Combine all the inputs into a single dictionary
    combined_info = {
        'user_info': user_info,
        'model_info': model_info,
        'feed_info': feed_info
    }

    # Convert the dictionary to a JSON string
    json_str = json.dumps(combined_info, indent=4)

    # Write the JSON string to a file
    with open(file_name, 'w') as file:
        file.write(json_str)

    return file_name
# Main function
def main():
    user_info, api_info, model_info, feed_info = setup_sidebar()
    answer = execute_model(api_info, model_info, feed_info) if st.sidebar.button("Execute") else ""
    st.markdown("#### LLM Output:")
    st.write(answer)

    #st.markdown("#### Inserted Data:")
    #st.write(feed_info['insert_data'])

    if st.sidebar.button("Export Inputs"):
        export_inputs_to_json(user_info,model_info, feed_info)
        st.success("Your inputs have been exported as JSON into your root depository. Please submit your market feed by sharing your JSON through the Github issue! We appreciate your contribution!")

if __name__ == "__main__":
    main()
