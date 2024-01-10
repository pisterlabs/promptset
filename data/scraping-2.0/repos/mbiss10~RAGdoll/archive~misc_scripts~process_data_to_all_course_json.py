from langchain.vectorstores import Chroma
import pickle
import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = API_KEY

DEV_SET_PATH = "./data/dev"
TEST_SET_PATH = "./data/test"

##########################################
# Extract Math/CS courses (JSON objects
# loaded as dicts) from courses.json file
##########################################


def get_course_list(dept_name):
    with open("./data/courses.json", "r") as f:
        course_dict = json.load(f)
        courses = course_dict["courses"]
        return [course for course in courses if course["department"] == dept_name]


cs_courses = get_course_list("CSCI")
math_courses = get_course_list("MATH")

print(f"Found {len(cs_courses)} CSCI courses.")
print(f"Found {len(math_courses)} MATH courses.")


course_header_groups = [
    [("descriptionSearch", "Course description")],
    [("gradingBasisDesc", "Grading basis")],
    [("classFormat", "Class format"), ("extraInfo", "Extra info"), ("crossListing",
                                                                    "Cross listing"), ("components", "Components"), ("departmentNotes", "Department notes")],
    [("courseAttributes", "Course attributes")],
    [("classReqEval", "Class requirements and evaluation")],
    [("prereqs", "Prerequisites")],
    [("enrolmentPreferences", "Enrolment preferences")]
]

section_headers = [("year", "Year"), ("semester", "Semester"), ("sectionType", "Section type"),
                   ("classType", "Class type"), ("meetings", "Meetings")]  # instructors added below



cs_course_objects = dict()
course_name_to_num_map = dict()
courses_processed = set()
dept_name = "CSCI"

for section in cs_courses:
    course_name = f"{section['department']} {section['number']}: {section['titleLong']}"

    # create source URL for this course using the course catalog
    course_num = section['number']
    course_id = section['courseID']
    course_url = f"https://catalog.williams.edu/{dept_name}/detail/?strm=&cn={course_num}&crsid={course_id}&req_year=0"

    # add info about this course if we haven't already
    if course_name not in courses_processed:
        course_name_to_num_map[section['titleLong']] = course_num
        course_obj = {
            # "year": section["year"],
            # "department": section["department"],
            "number": section["number"],
            "semesters": [section["semester"]],
            "titleLong": section["titleLong"],
            "titleShort": section["titleShort"],
            "description": section["descriptionSearch"],
            "url": course_url,
            "sections": [],
            "gradingBasisDesc": section["gradingBasisDesc"],
            "classFormat": section["classFormat"],
            "extraInfo": section["extraInfo"],
            "crossListing": section["crossListing"],
            "components": section["components"],
            "departmentNotes": section["departmentNotes"],
            "courseAttributes": section["courseAttributes"],
            "class_requirements_and_evaluation": section["classReqEval"],
            "prereqs": section["prereqs"],
            "enrolmentPreferences": section["enrolmentPreferences"]
        }
        cs_course_objects[course_name] = course_obj
        courses_processed.add(course_name)
    
    if section["semester"] not in cs_course_objects[course_name]["semesters"]:
        cs_course_objects[course_name]["semesters"].append(section["semester"])
    
    # add info about this section
    section_obj = {
        "section": section["section"],
        "classType": section["classType"],
        "sectionType": section["sectionType"],
        "instructors": [i["name"] for i in section['instructors']],
        "meetings": section["meetings"]
    }
    cs_course_objects[course_name]["sections"].append(section_obj)


final_json = {
    "mathematics_math_courses": [],
    "computer_science_csci_courses":
       cs_course_objects

}


with open(f"{DEV_SET_PATH}/temp/cs_ultra_json.json", "w") as f:
    f.write(json.dumps(final_json))

# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(cs_docs, embeddings)
# with open(f"{DEV_SET_PATH}/temp/db_cs_course_descriptions.pkl", "wb") as f:
#     f.write(pickle.dumps(db))

# vectordb = Chroma.from_documents(documents=cs_docs, embedding=embeddings,
#                                  persist_directory=f"{DEV_SET_PATH}/temp/cs_course_descriptions_chroma")
# vectordb.persist()


# cs_courses_objects = []
# math_courses_objects = []
# for result_list, course_list, dept_name in [(cs_courses_objects, cs_courses, "csci"), (math_courses_objects, math_courses, "math")]:
#     # log courses we've processed so we don't repeat course data across multiple sections
#     courses_processed = set()

#     for section in course_list:
#         section_name = f"{section['department']} {section['number']}: {section['titleLong']}, Section {section['section']}\n"
#         course_name = f"{section['department']} {section['number']}: {section['titleLong']}\n"

#         # create source URL for this course using the course catalog
#         course_num = section['number']
#         course_id = section['courseID']
#         course_url = f"https://catalog.williams.edu/{dept_name}/detail/?strm=&cn={course_num}&crsid={course_id}&req_year=0"

#         # add info about this course if we haven't already
#         if course_name not in courses_processed:
#             for header_group in course_header_groups:
#                 course_obj = {
#                     "course_name_full": course_name,
#                     "department": section["department"],
#                     "course_number": section["number"],
#                     "year": section["year"],
#                     "semester": section["semester"],
#                     **{header_parsed: section[header] for (header, header_parsed) in header_group if section[header]}
#                 }
#                 result_list.append(course_obj)
#                 courses_processed.add(course_name)

#         # add info about this *section* of the course
#         instructors = None
#         if "instructors" in section and section["instructors"] is not None:
#             # only include the instructor names, we don't care about IDs
#             instructors = [instructor["name"]
#                            for instructor in section["instructors"]]

#         section_obj = {
#             "section_name_full": section_name,
#             "section_number": section["section"],
#             "department": section["department"],
#             "course_number": section["number"],
#             "year": section["year"],
#             "semester": section["semester"],
#             "instructors": instructors,
#             **{segment_parsed: section[segment] for (segment, segment_parsed) in section_headers if section[segment]}
#         }

#         result_list.append(section_obj)


##########################################
# Add passages from the CS major requirements and
# department info PDF. This was manually extracted
# and cleaned up from the PDF. Cite the PDF as the
# source for these passages.
##########################################
# cs_major_documents = []
# with open(f"{DEV_SET_PATH}/cs_pdf_gold.txt", "r") as f:
#     for line in f:
#         cs_major_documents.append(Document(
#             page_content=line,
#             metadata={"source_url": "https://catalog.williams.edu/pdf/csci.pdf"}))

# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(cs_major_documents, embeddings)
# with open(f"{DEV_SET_PATH}/temp/db_cs_major.pkl", "wb") as f:
#     f.write(pickle.dumps(db))

# # Do the same for math
# with open(f"{TEST_SET_PATH}/math_pdf_gold.txt", "r") as f:
#     for line in f:
#         math_data.append((line, "https://catalog.williams.edu/pdf/math.pdf"))


# print("Data parsing complete.")
# print(f"Total CSCI passages created: {len(cs_data)}")
# print(f"Total MATH passages created: {len(math_data)}")


##########################################
# Save data to a file for easy viewing
##########################################
# def write_to_json(outpath, obj_list):
#     with open(outpath, "w") as f:
#         out = json.dumps(obj_list)
#         f.write(out)


# write_to_json(
#     f"{DEV_SET_PATH}/cs_courses_objects.json", cs_courses_objects)
# write_document_list_to_json(
#     f"{DEV_SET_PATH}/cs_major_documents.json", cs_major_documents)


##########################################
# Create vectorstores from the data, then
# persist the vectorstores to disk.
##########################################

# 🔴 Uncomment the lines below to write the new vectorestore db   🔴
# 🔴 This may cause the model's behavior to change since it will  🔴
# 🔴 create a new knowledge store. It also costs API credits.     🔴

# embeddings = OpenAIEmbeddings()

# cs_major_vectorstore = Chroma.from_documents(
#     cs_major_documents,
#     embeddings,
#     persist_directory=f"{DEV_SET_PATH}/cs_major_vectorstore")
# # persist the db
# cs_major_vectorstore.persist()


# cs_courses_vectorstore = Chroma.from_documents(
#     cs_courses_documents,
#     embeddings,
#     persist_directory=f"{DEV_SET_PATH}/cs_courses_vectorstore")
# # persist the db
# cs_courses_vectorstore.persist()
