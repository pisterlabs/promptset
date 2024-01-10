import streamlit as st
import openai
import os
# import json
import time


from dotenv import load_dotenv, find_dotenv
from PIL import Image


_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]


def input_dashboard():
    project_input = {}

    project_input["category"] = st.text_input(
        "Category of your project",
        "Example: Software, Home Renovation, Ad Campaign & ...",
    )
    project_input["subject"] = st.text_area(
        "- Subject of your project",
        "Example: Designing website sitemap for my consulting company and creating wireframe, designing UI/UX, and deploying on wix.com",
    )

    project_input["budget"] = st.number_input("Enter your desired budget in USD", 1000)
    project_input["financial_constraints"] = st.text_area(
        "Enter your financial constraints",
        "Example: I can pay 50% upfront and 50% after the project is completed",
    )

    project_input["duration"] = st.number_input(
        "Enter your desired duration in days", 30
    )
    project_input["schedule_constraints"] = st.text_area(
        "Enter your schedule constraints",
        "Example: I need this project to be completed in 2 weeks starting from September 1st 2023",
    )

    project_input["comment"] = st.text_area("Any additional comments?")

    return project_input


input_dict = {
    "category": "Software",
    "subject": "Designing website sitemap for my consulting company and creating wireframe, designing UI/UX, and deploying on wix.com",
    "budget": 1000,
    "financial_constraints": "I can pay 50% upfront and 50% after the project is completed",
    "duration": 14,
    "schedule_constraints": "I need this project to be completed in 2 weeks starting from September 1st 2023",
    "comment": "I need this project to be completed in 2 weeks starting from September 1st 2023",
}


def get_plan(input_dict):
    response = openai.chat.completions.create(
        # model="gpt-3.5-turbo-16k",
        model='gpt-4-1106-preview',
        messages=[
            {
                "role": "system",
                "content": "You are a WBS expert. Your task is to generate a consistent two-level deep Work Breakdown Structure (WBS) table based on the given project attributes. The table should be in markdown format suitable for Streamlit. The columns should be 'WBS Number', 'WBS Activity', 'Cost', & 'Duration'. Ensure the WBS numbering is hierarchical (e.g., 1, 1.1, 1.2, 2, 2.1, etc.). Make sure to factor in the provided constraints and comments when generating the WBS. Your output should be reliable and should not vary significantly upon repeated requests.",
            },
            {
                "role": "user",
                "content": f"I'm planning a project with the following attributes:\n1. Category: {input_dict['category']}\n2. Subject: {input_dict['subject']}\n3. Budget: ${input_dict['budget']}\n4. Financial Constraints: {input_dict['financial_constraints']}\n5. Duration: {input_dict['duration']} days\n6. Schedule Constraints: {input_dict['schedule_constraints']}\n7. Comment: {input_dict['comment']}\nPlease generate a WBS table based on these details.",
            },
            {
                "role": "assistant",
                "content": "Sure! Here's a two-level deep WBS table for your project:\n\n| WBS Number | WBS Activity | Cost | Duration |\n|------------|--------------|------|----------|\n| 1 | Main Activity 1 | $XXXX | XX days |\n| 1.1 | Sub-Activity 1.1 | $XXXX | XX days |\n| 1.2 | Sub-Activity 1.2 | $XXXX | XX days |\n| 2 | Main Activity 2 | $XXXX | XX days |\n| 2.1 | Sub-Activity 2.1 | $XXXX | XX days |\n... \n\nEnsure you review and adjust the generated activities, costs, and durations as per the specifics of your project.",
            },
        ],
    )

    return response.choices[0].message.content


def main():
    # image = Image.open("/Users/kvmmn/Desktop/ai.toyon/genx/plannule/banner.png")
    image = Image.open("banner.png")
    st.image(image, use_column_width=True)
    st.title("Plannule")
    st.text("Project Management Made Easy")
    st.divider()

    input_dict = input_dashboard()

    if st.button("Submit"):
        progress_text = "Planning & curation in progress... Please wait :)"
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)

        st.write("Here is your curated plan:")
        output_plan_json = get_plan(input_dict)
        st.markdown(output_plan_json)
        st.write("Finished!")


if __name__ == "__main__":
    main()
