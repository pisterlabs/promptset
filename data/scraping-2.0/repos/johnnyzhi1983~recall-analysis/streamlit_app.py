import pandas as pd
import streamlit as st
import sqlite3
st.set_page_config(page_title='RecallAnalysis', layout='wide')
import openai


st.title("Recall Analysis")

openai.api_key = st.secrets["openAI_SK"]


def get_data_from_DB() -> pd.DataFrame:
  con = sqlite3.connect('recall.db')
  df = pd.read_sql_query("SELECT url, title, date, content FROM tb_recalls", con)
  con.close()
  return df


if __name__ == '__main__':
  recall_exp = st.expander("Click to see the raw data...", expanded=False)
  with recall_exp:
    df = get_data_from_DB()
    st.dataframe(df)
