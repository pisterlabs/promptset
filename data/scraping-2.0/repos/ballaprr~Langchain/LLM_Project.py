from dotenv import load_dotenv
from langchain import OpenAI
import streamlit as st
from langchain.agents import create_csv_agent, initialize_agent, Tool
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, download_loader
from pathlib import Path
import openai
import pandas as pd
import requests
import re
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

os.environ["OPENAI_API_KEY"] = api_key


st.set_page_config(page_title="Baseball CSV")
st .header("Prompt: ")


year_options = [str(year) for year in range(1876, 2024)] # list of years
selected_years = []
selected_years = st.multiselect('Select Years', year_options, default=selected_years, key='multiselect_1') # years that are selected

def pull_baseball_data(selected_years, iterate, count):
    merged_df = pd.DataFrame()
    for year in selected_years:
        if url_st == 'player pitching':
            iterate += 1
            player_pitching_url = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player?stitch_env=prod&season=" + year + "&sportId=1&stats=season&group=pitching&gameType=R&limit=120&offset=0&sortStat=earnedRunAverage&order=asc"
            r_player_pitching = requests.get(url=player_pitching_url).json()
            values = pd.DataFrame(r_player_pitching['stats']).values.tolist()
            df = pd.DataFrame(values, columns = list(r_player_pitching['stats'][0].keys()))
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if (count == iterate):
                merged_df[['year', 'playerFullName', 'teamName', 'position', 'leagueName', 'era', 'holds', 'whip', 'outs']].to_csv("_".join(map(str, selected_years)) + "_data_player_pitch.csv", index=False)
        elif url_st == 'player hitting':
            iterate += 1
            player_hitting_url = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player?stitch_env=prod&season=" + year + "&sportId=1&stats=season&group=hitting&gameType=R&limit=30&offset=0&sortStat=onBasePlusSlugging&order=desc"
            r_player_hitting = requests.get(url=player_hitting_url).json()
            values = pd.DataFrame(r_player_hitting['stats']).values.tolist()
            df = pd.DataFrame(values, columns = list(r_player_hitting['stats'][0].keys()))
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if (count == iterate):
                merged_df[['year', 'playerFullName', 'teamName', 'position', 'leagueName', 'strikeoutsPerPlateAppearance', 'runs', 'doubles', 'triples', 'rbi']].to_csv("_".join(map(str, selected_years)) + "_data_player_hitt.csv", index=False)
        elif url_st == 'team hitting':
            iterate += 1
            team_hitting_url = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/team?stitch_env=prod&sportId=1&gameType=R&group=hitting&order=desc&sortStat=onBasePlusSlugging&stats=season&season=" + year + "&limit=30&offset=0"
            r_team_hitting = requests.get(url=team_hitting_url).json()
            values = pd.DataFrame(r_team_hitting['stats']).values.tolist()
            df = pd.DataFrame(values, columns = list(r_team_hitting['stats'][0].keys()))
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if (count == iterate):
                merged_df[['year', 'teamName', 'leagueName', 'runs', 'doubles', 'triples', 'homeRuns', 'hits', 'rbi']].to_csv("_".join(map(str, selected_years)) + "_data_team_hitt.csv", index=False)
        elif url_st == 'team pitching':
            iterate += 1
            team_pitching_url = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/team?stitch_env=prod&sportId=1&gameType=R&group=pitching&order=asc&sortStat=earnedRunAverage&stats=season&season=" + year + "&limit=30&offset=0"
            r_team_pitching = requests.get(url=team_pitching_url).json()
            values = pd.DataFrame(r_team_pitching['stats']).values.tolist()
            df = pd.DataFrame(values, columns = list(r_team_pitching['stats'][0].keys()))
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if (count == iterate):
                merged_df[['year', 'teamName', 'leagueName', 'strikeOuts', 'era', 'inningsPitched', 'holds', 'whip', 'battersFaced', 'balks']].to_csv("_".join(map(str, selected_years)) + "_data_team_pitch.csv", index=False)


st.write("Selected Years: ", selected_years)

url = ['player pitching', 'player hitting', 'team pitching', 'team hitting']

url_st = st.selectbox('Select data for analysis', url)
print(url_st)
if url_st not in url:
    st.warning("Please select a valid analysis option.")
    st.stop()
iterate = 0
count = len(selected_years)

# Initialize process_button to False
process_button = False

# Create a button to toggle the value of process_button
download_button = st.button("Download csv file")

if download_button:
    process_button = True  # Toggle the value of process_button

while process_button:
    pull_baseball_data(selected_years, iterate, count)
    if download_button:
        process_button = False




user_csv=st.file_uploader("Upload your CSV",type=["csv"])

def save_txt_content(content, filename):
    # Save the TXT content to a file
    with open(filename, "w") as txt_file:
        txt_file.write(content)


if user_csv is not None:

    csv_reader = download_loader("SimpleCSVReader")
    csv_file = csv_reader(encoding="utf-8")
    print(user_csv.name)
    docs = csv_file.load_data(file=Path(user_csv.name))

    query_engine = GPTVectorStoreIndex.from_documents(docs).as_query_engine()
    response = query_engine.query("Summarize the data")
    st.write(response.response)

    user_input = st.text_input("Ask about data: ")
    response_custom = query_engine.query(user_input)
    st.write(response_custom.response)


    user_input_data =st.text_input("Ask a question about your CSV: ")

    llm = OpenAI(temperature = 0, model_name="text-davinci-003")
    csv = create_csv_agent(llm, user_csv.name, verbose=True)
    res = csv.run(user_input_data)

    st.write(res)

    txt_content = f"Year:\n{selected_years}\n"
    txt_content += f"Data\n{url_st}"
    txt_content += f"Summary\n{response.response}"
    txt_content += f"Question about data\n{user_input}\n{response_custom.response}\n"
    txt_content += f"Question about your CSV\n{user_input_data}\n{res}\n"

    if st.button("Download Response as txt"):
        save_txt_content(txt_content, "responses.txt")
        st.success("TXT file saved!")

