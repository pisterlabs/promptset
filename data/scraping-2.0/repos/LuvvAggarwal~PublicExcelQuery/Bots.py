import pandas as pd
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
load_dotenv()

# --------- MERGE 

# df1 = pd.read_csv("Coursera_reviews.csv")
# df2 = pd.read_csv("Coursera_courses.csv")
# merged = df1.merge(df2, on="course_id", how="outer").fillna("")
# merged.to_csv("merged.csv")

# -----------------------------------------------------
memory = ConversationBufferMemory()
# df = pd.read_pickle("./Coursera_merged.pkl",compression='bz2')
# ques_array = ["Total count of reviews for course where course id is financial-markets-global", "Summarize reviews in 2 sentences each of pros and cons for course where course id is financial-markets-global"]
# for q in ques_array:
#     print(q)
def runExcelQuery(ques, df):
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    return agent.run(ques)
