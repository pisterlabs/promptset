"""
Takes as input a courses.json file with info about all course sections, and a "gold standard"
.txt copy of the PDF with major info for a given department. 

Outputs two pickled vectorstores per department: one containing Documents without embeddings
to be used for TF-IDF retrieval (these are stored at CS_DATA_OUT_PATH and MATH_DATA_OUT_PATH),
and the other containing embedded Documents to be used for FAISS retrieval (these are stored
at CS_VECTORSTORE_OUT_PATH and MATH_VECTORSTORE_OUT_PATH).
"""

import pickle
import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = API_KEY

# directory paths
DEV_SET_PATH = "./data/dev"
TEST_SET_PATH = "./data/test"

# Paths to store pickled list of documents.
# These are used by the tf-idf retriever
CS_DATA_OUT_PATH = f"{DEV_SET_PATH}/cs_data.pkl"
MATH_DATA_OUT_PATH = f"{TEST_SET_PATH}/math_data.pkl"

# Paths to store pickled vectorstores of embeddings.
# These are used by the FAISS vectorstore retriever.
# Set path to None if you do not want to create/store a vectorstore.
# for either CS or math
CS_VECTORSTORE_OUT_PATH = None  #f"{DEV_SET_PATH}/db_cs_with_sources.pkl"
MATH_VECTORSTORE_OUT_PATH = f"{TEST_SET_PATH}/db_math_with_sources.pkl"


if __name__ == "__main__":
    ##########################################
    # Extract Math/CS courses (JSON objects
    # loaded as dicts) from courses.json file
    ##########################################
    cs_courses = None
    math_courses = None
    with open("./data/courses.json", "r") as f:
        course_dict = json.load(f)
        courses = course_dict["courses"]

        cs_courses = [
            course for course in courses if course["department"] == "CSCI"]

        math_courses = [
            course for course in courses if course["department"] == "MATH"]

        print(f"Found {len(cs_courses)} CSCI courses.")
        print(f"Found {len(math_courses)} MATH courses.")


    ##########################################
    # Create list of tuples of the form (passage, source)
    # where passage is some text about a course or section
    # and source is a URL.
    ##########################################
    course_header_groups = [
        ["descriptionSearch"],
        ["gradingBasisDesc"],
        ["classFormat", "extraInfo", "crossListing", "components", "departmentNotes"],
        ["courseAttributes"],
        ["classReqEval"],
        ["prereqs"],
        ["enrolmentPreferences"]
    ]

    section_headers = ["year", "semester", "sectionType",
                    "classType", "instructors", "meetings"]

    # list of tuples of the form (passage, source)
    cs_data = []
    math_data = []
    for data, course_list, dept_name in [(cs_data, cs_courses, "csci"), (math_data, math_courses, "math")]:
        # log courses we've processed so we don't repeat course data across multiple sections
        courses_processed = set()

        for section in course_list:
            section_name = f"{section['department']} {section['number']} ({section['titleLong']}) Section {section['section']}"
            course_name = f"{section['department']} {section['number']} ({section['titleLong']})"

            # create source URL for this course using the course catalog
            course_num = section['number']
            course_id = section['courseID']
            course_url = f"https://catalog.williams.edu/{dept_name}/detail/?strm=&cn={course_num}&crsid={course_id}&req_year=0"

            # add info about this course if we haven't already
            if course_name not in courses_processed:
                for header_group in course_header_groups:
                    course_line = f"{course_name} "
                    course_line += ", ".join(
                        [f"{header}: {section[header]}" for header in header_group])
                    course_line += "\n"
                    data.append((course_line, course_url))
                    courses_processed.add(course_name)

            # add info about this section of the course
            if "instructors" in section and section["instructors"] is not None:
                for instructor in section["instructors"]:
                    del instructor["id"]

            section_line = f"{section_name} "
            section_line += ", ".join(
                [f"{segment}: {section[segment]}" for segment in section_headers])
            section_line += "\n"
            data.append((section_line, course_url))


    ##########################################
    # Add passages from the CS major requirements and
    # department info PDF. This was manually extracted
    # and cleaned up from the PDF. Cite the PDF as the
    # source for these passages.
    ##########################################
    with open(f"{DEV_SET_PATH}/cs_pdf_gold.txt", "r") as f:
        for line in f:
            cs_data.append((line, "https://catalog.williams.edu/pdf/csci.pdf"))

    # Do the same for math
    with open(f"{TEST_SET_PATH}/math_pdf_gold.txt", "r") as f:
        for line in f:
            math_data.append((line, "https://catalog.williams.edu/pdf/math.pdf"))


    print("Data parsing complete.")
    print(f"Total CSCI passages created: {len(cs_data)}")
    print(f"Total MATH passages created: {len(math_data)}")


    ##########################################
    # Save data to a file for easy viewing
    ##########################################
    if CS_DATA_OUT_PATH:
        with open(CS_DATA_OUT_PATH, "wb") as f:
            pickle.dump(cs_data, f)
            print(f"Saved pickled CSCI data to {CS_DATA_OUT_PATH}")

    if MATH_DATA_OUT_PATH:
        with open(MATH_DATA_OUT_PATH, "wb") as f:
            pickle.dump(math_data, f)
            print(f"Saved pickled MATH data to {MATH_DATA_OUT_PATH}")


    ##########################################
    # Create a vectorstore from the data.
    # Serialize and store it.
    ##########################################
    cs_texts, cs_sources = list(zip(*cs_data))
    math_texts, math_sources = list(zip(*math_data))

    # 🔴 Uncomment the lines below to write the new vectorestore db   🔴
    # 🔴 This may cause the model's behavior to change since it will  🔴
    # 🔴 create a new knowledge store. It also costs API credits.     🔴

    embeddings = OpenAIEmbeddings()

    for (outpath, texts, sources) in [(CS_VECTORSTORE_OUT_PATH, cs_texts, cs_sources), (MATH_VECTORSTORE_OUT_PATH, math_texts, math_sources)]:
        if outpath:
            db = FAISS.from_texts(texts, embeddings, metadatas=[
                                {"source": source} for source in sources])
            
            with open(outpath, "wb") as f:
                pickle.dump(db, f)
                print(f"Saved pickled vectorstore to {outpath}")
