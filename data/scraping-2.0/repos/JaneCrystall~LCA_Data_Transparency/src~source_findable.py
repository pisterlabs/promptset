import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from retrying import retry

load_dotenv()
openai_client = OpenAI()


@retry(wait_fixed=300, stop_max_attempt_number=5)
def findability(query: str):
    completion = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.0,
        seed=111,
        messages=[
            {
                "role": "system",
                "content": """Based on the source information user provided, examine source findability according to the following criteria. Return "findable" if the source is findable, and "unfindable" if the source is unfindable. 
                And if the source is unfindable, please provide the reason why it is unfindable.

Article in Periodical:
If the source is an article in a periodical, verify whether the source information includes a DOI number or both the article title and the author(s). If either the DOI is present or both the article title and author(s) are available, then the source is findable. If neither is available, the source is unfindable. 

Chapter in Anthology:
If the source is a chapter in an anthology, check if the source information contains the chapter title, author(s), book title, and either the year of publication or the edition version. If all these elements are present, the source is findable. If any are missing, the source is unfindable.

Monograph:
For a monograph source, determine if the title, author(s), and either the year of publication or the edition version are included. If these details are provided, the source is findable. If not, the source is unfindable.

Industry Report:
If the source is an industry report, check for the presence of the report title, the authoring organization, and the year of publication. If all are present, the source is findable. If any detail is absent, the source is unfindable.

Standard:
Verify if the source as a standard includes either the standard number or both the title and the issuing organization's name along with the publication time. If either criterion is met, the source is findable. If both criteria are not met, the source is unfindable.

Environmental Product Declaration (EPD):
If the source is an EPD, ensure it has either the declaration number or the combination of product name, declaration holder, publisher, and issue time. If one of these sets is complete, the source is findable. If not, the source is unfindable.

Patent:
Check if the patent source has either a patent number or the combination of title, author, and time of issue. If one of these sets of information is present, the source is findable. If neither set is complete, the source is unfindable.

Statistical Documents:
For statistical documents, confirm if the document title, authoring organization, and year of publication are present. If these elements are there, the source is findable. If not, it is unfindable.

Software or Database:
Examine if the software or database source has the name and either the version or release year. If both pieces of information are available, the source is findable. If any is missing, then the source is unfindable.

Personal Communication:
If the source is from personal communication, verify the inclusion of the correspondent's name and the date of communication. If both are documented, the source is findable. If either is absent, the source is unfindable.

Direct Measurement:
Ensure that the direct measurement source includes the date of measurement and the location of measurement. If both are present, the source is findable. If not, the source is unfindable.

Website:
For a website source, check if it includes the URL and the date retrieved. If both are provided, the source is findable. If either is missing, the source is unfindable.
""",
            },
            {"role": "user", "content": query},
        ],
    )

    message = completion.choices[0].message.content
    return message


file_path = "Gabi_source_category.xlsx"

# 读取Excel文件
df = pd.read_excel(file_path)
source_info = df["source_info"]
for index, source in enumerate(source_info):
    source_findability = findability(source)
    df.loc[df["source_info"] == source, "source_findability"] = source_findability
    print(index + 1, source_findability)


df.to_excel("Gabi_source_findability.xlsx", index=False)
