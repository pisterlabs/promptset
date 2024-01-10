# Importing the necessary libraries

import json
import re
import time
from pprint import pprint
import argparse
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

tqdm.pandas()


def initialize_api() -> OpenAI:
    """Initializing the OpenAI API object"""
    client = OpenAI()

    return client


def naming_convention() -> str:
    """Name of the final output HTML file"""
    name = input("What would you like to name your html file? ")

    return name


def get_user_input() -> [str]:
    """Gathering some inputs from users to be used as prompts"""
    parser = argparse.ArgumentParser(description="SOP Guideline Generator")

    parser.add_argument(
        "objective", type=str, help="Objective or purpose of the guideline"
    )
    parser.add_argument("author", type=str, help="Author of the guideline")
    parser.add_argument("audience", type=str, help="Intended audience of the guideline")
    parser.add_argument("format", type=str, help="Preferred format of the guideline")
    parser.add_argument(
        "instructions", type=str, help="Additional details for each step"
    )

    args = parser.parse_args()

    return [args.objective, args.author, args.audience, args.format, args.instructions]


def get_message_memory(
    new_question: str,
    previous_repsonse: str,
    system_context: str,
    client: OpenAI,
    model_id: str,
) -> dict:
    """helper function to run chat conversations"""
    if previous_repsonse is not None:
        response = client.chat.completions.create(
            model=model_id,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": new_question},
                {"role": "assistant", "content": previous_repsonse},
            ],
            temperature=0,
            seed=42,
        )
    else:
        response = client.chat.completions.create(
            model=model_id,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": new_question},
            ],
            temperature=0,
            seed=42,
        )

    return response.choices[0].message.content


def run_chat_conversations(
    client: OpenAI, model_id: str, context: str, messages: str
) -> dict:
    """running the multi-step conversation"""
    last_response = None
    for user_input in tqdm(messages):
        chat_response = get_message_memory(
            user_input, last_response, context, client, model_id
        )
        last_response = chat_response

    return json.loads(last_response)


def get_images(image_description: str, client: OpenAI) -> str:
    """helper function to generate the images from a given prompt"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    time.sleep(5)

    return response.data[0].url


def generate_images_and_update_df(df, client) -> pd.DataFrame:
    """generating the images and including them into the df"""
    df["image_prompt"] = df[df.columns.to_list()[1:]].apply(
        lambda x: "Generate image where "
        + x[0]
        + ". "
        + x[1]
        + ". Name the image as "
        + x[2],
        axis=1,
    )

    df["Example"] = df["image_prompt"].progress_apply(get_images, client=client)

    return df


def path_to_image_html(path: str) -> str:
    """converting image urls into html tags"""
    return '<img src="' + path + '" width="200" >'


def main():
    """main function to execute the entire process of generating an SOP"""
    # Initiallizing the openai api object
    client = initialize_api()

    # Defining the variables to be used as inputs in model
    model_id = "gpt-4-1106-preview"
    model_context = "You are an expert Standard Operating Procedure (SOP) generator in a JSON table format.\n \
        Make sure that the instructions given are clear and comprehensive, good enough to generate images as well."
    user_inputs = get_user_input()

    # Using the given inputs from users, we run them as a loop into the chat function - multi-conversation
    model_responses = run_chat_conversations(
        client, model_id, model_context, user_inputs
    )

    pprint(model_responses)

    # Converting the JSON response for the instructions into a flattened pandas dataframe
    try:
        df = json_normalize(model_responses, record_path=["SOP_Table", "Procedure"])

    except KeyError as ke:
        print(f"KeyError: {ke}. Expected Key not found in the JSON response")
        df = json_normalize(
            model_responses, record_path=["SOP_Table", "Procedure_Steps"]
        )

    except Exception as e:
        print(f"An unexpected error has occured: {e}")

    # Now to generate images and update the dataframe
    df = generate_images_and_update_df(df, client)
    df.drop(columns=["image_prompt"], inplace=True)

    # Convert dataframe into html
    df_html = df.to_html(
        escape=False, formatters=dict(Example=path_to_image_html), index=False
    )

    # Next we want to convert the JSON response for the summary portion into a flattened pandas dataframe
    try:
        summary = json_normalize(model_responses)
        summary.drop(columns="SOP_Table.Procedure", inplace=True)

    except KeyError as ke:
        print(f"KeyError: {ke}. Expected Key not found in the JSON response")
        summary = json_normalize(model_responses)
        summary.drop(columns="SOP_Table.Procedure_Steps", inplace=True)

    except Exception as e:
        print(f"An unexpected error has occured - Key not found in summary: {e}")

    # Rename summary columns and convert df to html
    summary.columns = [re.sub("^(.*\.)", "", col) for col in summary.columns]
    summary_html = summary.to_html(index=False)

    # Now to merge both summary html and instruction html in one html page - insert gap between both tables

    # Merge HTML content with a gap between tables
    merged_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .table-container {{
                margin-bottom: 50px;  /* Adjust the gap between tables */
            }}

            table {{
                border-collapse: collapse;
                width: 50%;
                margin: 20px;
            }}
            th, td {{
                border: 1px solid black;
                padding: 20px;
                text-align: left;
            }}
            h2 {{
                color: blue;
            }}

            body {{ background-color: #e0e0e0; }}
        </style>
    </head>
    <body>
        <h2>Responsibilities and Governance</h2>
            {summary_html}
        <h2>Instructions and Guidelines</h2>
            {df_html}
        </div>
    </body>
    </html>
    """

    # Save the merged HTML content into a new file
    with open(f"{naming_convention()}.html", "w", encoding="utf-8") as merged_file:
        merged_file.write(merged_html_content)


if __name__ == "__main__":
    main()
