"""
THIS FILE: is used to create Question and Answer table for RAGAS that has the columns: context, question, answer, and summary. 
The purpose is to create an automated ground truth from the dump file of the wikipedia articles (from sustainability methods wiki).
At the end, it outputs JSON file that we can use for RAGAS evaluation. 
"""

import json
from rag_llms import load_llm_gpt4,load_llm_gpt35

import openai
import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

SEGMENTED_CSV_PATH = "data/wiki_checklist_segmented.csv"
SEGMENTED_CSV_PATH_NEW = "data/wiki_checklist_segmented_new.csv"
csv_dump_path = "data/Sustainability+Methods_dump_checklist.csv"
GROUND_TRUTH_JSON_PATH = "data/ground_truth.json"



MODEL_ID = "gpt-3.5-turbo"
# MODEL_ID = "gpt-4"

ground_truth_json_schema = {
    "contexts": "part of the article or paragraphs that you want to summarize",
    "summary": "summary of the context(s)",
    "question": "question that you want to ask about the context(s)",
    "ground_truths": "answer from the human",
}




generate_ground_truth_template = str('''
The wikipedia title is: {context}
The wikipedia article is: {article}

----------------------------------------------
You are given a wikipedia title and article. 
You are a world class algorithm for summarizing and extracting information in structured formats. Do not answer anything else than the JSON formatted answer. 

Do these steps to the article above:

1. Split the article into paragraphs. I'd rename that as "context". Take only the paragraphs that consisted of a minimum of 5 sentences in the article. 
2. Create a summary of each paragraph using only about half of the length of the initial paragraph. 
3. Read the summary, or use the summary as an input. Ask a general "question" by reading the summary using either what, who, when, how, or why. Do not ask question by saying "in this article", "in this paragraph", or "as mentioned here". The question must be a general question!
4. Read the question and Answer your own question based on the summary. It is imperative that you do not add additional information!
5. After you have answer it, check your answer by comparing it with the original paragraph from the article. Continue to the next step only when the answer is correct. If you cannot answer, then redo the step 2, 3, and 4. By the way, I will refer the answer that passed this test as "ground_truth" from now on.
6. Provide them in JSON format with the following keys: context, summary, question, ground_truths. Provide the full context according to the original paragraph. Important that you pay attention to the double quotes ", since in JSON file, the value should NOT include double quotes. If you must state a double quote in the value, you have to put backslash before it like this \" in order to escape it. 

Example answer: 
    [
        {{
            "contexts" : "part of the article or paragraphs that you want to summarize",
            "summary" : "summary of the context(s)",
            "question" : "question that you want to ask about the context(s)",
            "ground_truths" : "<answer of the question>"
        }},
        {{
            "contexts" : "part of the article or paragraphs that you want to summarize",
            "summary" : "summary of the context(s)",
            "question" : "question that you want to ask about the context(s)",
            "ground_truths" : "<answer of the question>"
        }}, 
        {{
             continue until the last paragraph
        }}
    ]


Don't answer anything else than the JSON formatted answer.
'''
)


# read csv to dataframe
sustmethods_df  = pd.read_csv(csv_dump_path, delimiter=";")


##############
def split_long_text_into_chunks(long_text):

    # Split the text into paragraphs based on '\n\n'
    paragraphs = long_text.split('\n\n')

    # Initialize variables
    chunk = ""
    chunks = []
    word_count = 0

    # Loop through each paragraph to construct chunks
    for paragraph in paragraphs:
        # Count the number of words in the paragraph
        num_words = len(paragraph.split())

        # Check if adding this paragraph would exceed the 1,500-word limit
        if word_count + num_words > 1000:
            # Save the current chunk and start a new one
            chunks.append(chunk)
            chunk = ""
            word_count = 0

        # Add the paragraph to the current chunk and update the word count
        if chunk:
            chunk += "\n\n"  # Add a separator between paragraphs
        chunk += paragraph
        word_count += num_words

    # Add the last chunk if it's not empty
    if chunk:
        chunks.append(chunk)

    # 'chunks' now contains the text segmented into chunks of approximately 1,500 words each, 
    # with each chunk ending at the end of a paragraph.
    # print(chunks)
    return chunks

###############
def create_segmented_df(wiki_df):
    # # Your original DataFrame (for demonstration)
    # wiki_df = pd.DataFrame({
    #     'title': ['title1', 'title2'],
    #     'text': ["...Your Wikipedia Text Here...", "...Another Wikipedia Text..."]
    # })

    # Initialize an empty DataFrame to store the new chunks
    segmented_df = pd.DataFrame(columns=['title', 'segmented_text', 'status'])

    # Iterate over each row in the original DataFrame
    for index, row in wiki_df.iterrows():
        title = row['title']
        text = row['text']
        status = row['status']
        
        # Use the previously defined function to split the text into chunks
        # if text's length is less than 100, skip
        if len(str(text)) < 100:
            continue
        chunks = split_long_text_into_chunks(text)
        print("Chunks:", chunks)

        
        # Populate the new DataFrame
        for chunk in chunks:
            # print("Current chunk:", chunk)
            new_row = {'title': title, 'segmented_text': chunk, 'status': status}
            segmented_df = pd.concat([segmented_df, pd.DataFrame([new_row])], ignore_index=True)

    # 'segmented_df' now contains the segmented text with their original titles
    print(segmented_df)
    return segmented_df


def create_segmented_df_csv(segmented_df, csv_path = SEGMENTED_CSV_PATH):
    # save segmented_df to csv
    segmented_df.to_csv(csv_path, index=False, sep=";")
    print("Saved segmented_df to csv")

# segmented_df = create_segmented_df(sustmethods_df)
# create_segmented_df_csv(segmented_df)

###############################################################################

def append_raw_json_to_file(file_path, json_object):
    raw_json_str = json.dumps(json_object, indent=4)[1:-1]  # Serialize and remove '[' and ']'
    with open(file_path, 'ab') as f:
        f.write(raw_json_str.encode("utf-8"))

def create_ground_truth_json(segmented_df, llm):
    # for every row in the segmented_df
    # generate the ground truth using openai 

    prompt = PromptTemplate(
        input_variables=["context", "article"],
        template=generate_ground_truth_template,
    )

    # Ensure the file exists; create an empty one if not
    if not os.path.exists(GROUND_TRUTH_JSON_PATH):
        with open(GROUND_TRUTH_JSON_PATH, "w") as f:
            f.write("[]")

    # Create new csv file to keep track which article has been processed
    if not os.path.exists(SEGMENTED_CSV_PATH_NEW):
        with open(SEGMENTED_CSV_PATH_NEW, "w") as f:
            f.write("title;text;status\n")
    
    chain = LLMChain(llm = llm, prompt=prompt)

    # call response for every row in the segmented_df
    for index, row in segmented_df.iterrows():
        if row["status"] == "done" or row["status"] == "skip":
            continue

        if len(str(row["segmented_text"]))<100:
            continue

        response = chain.run({
            "context": row["title"],
            "article": row["segmented_text"],
        })
        print(response)
        try: 
            json_response = json.loads(response)
        except:
            json_response = response
            print("response is not json, check the response, probably need more token")

        # Append the raw JSON string to the file
        # append_raw_json(GROUND_TRUTH_JSON_PATH, response)
        append_raw_json_to_file(GROUND_TRUTH_JSON_PATH, json_response)
        print("Saved ground truth to json")

        # update ground truth csv status to "done"
        row["status"] = "done"

        # append the newly updated row to the new csv
        segmented_df_new = pd.DataFrame([row])
        segmented_df_new.to_csv(SEGMENTED_CSV_PATH_NEW, mode='a', header=False, index=False, sep=";")
        print("Updated segmented_df_new csv")


segmented_df = pd.read_csv(SEGMENTED_CSV_PATH, delimiter=";")
# segmented_df_row= segmented_df.iloc[116:118]
llm = load_llm_gpt4()
create_ground_truth_json(segmented_df, llm)

# append_raw_json(GROUND_TRUTH_JSON_PATH, "test")


