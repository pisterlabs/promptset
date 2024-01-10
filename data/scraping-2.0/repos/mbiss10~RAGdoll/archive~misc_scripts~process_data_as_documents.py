"""
Takes as input the courses.json file and the gold standard CS major requirements/department txt file
(with lines pulled from the PDF).

Writes to the following 4 files:
-   cs_courses_documents.pkl: a pickled vectorstore created from a list of Document objects for each course
    AND section in the CS department. Course information is more general than section information, but there
    are documents for both.
    
-   cs_courses_documents.json: a JSON file containing the same information as above, but in JSON format.

-   cs_major_documents.pkl: a pickled vectorstore created from a list of Document objects for lines from
    the CS major requirements gold standard file.

-   cs_major_documents.json: a JSON file containing the same information as above, but in JSON format.
"""

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


#####################################################
# Create list of Document objects whose page_content
# is some info about a course or section and whose
# metadata is data that can be filtered on (e.g. year)
####################################################

course_header_groups = [
    # The first titem in each tuple is the key to get the data from the JSON object,
    # and the second is a more readable name that's used in the document object
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


cs_courses_documents = []
math_courses_documents = []
for result_list, course_list, dept_name in [(cs_courses_documents, cs_courses, "csci"), (math_courses_documents, math_courses, "math")]:
    # log courses we've processed so we don't repeat course data across multiple sections
    courses_processed = set()

    for section in course_list:
        section_name = f"{section['department']} {section['number']}: {section['titleLong']}, Section {section['section']}\n"
        course_name = f"{section['department']} {section['number']}: {section['titleLong']}\n"

        # create source URL for this course using the course catalog
        course_num = section['number']
        course_id = section['courseID']
        course_url = f"https://catalog.williams.edu/{dept_name}/detail/?strm=&cn={course_num}&crsid={course_id}&req_year=0"

        # add info about this course if we haven't already
        if course_name not in courses_processed:
            for header_group in course_header_groups:
                course_content = f"{course_name} "
                course_content += ", ".join(
                    [f"{header_parsed}: {section[header]}" for (header, header_parsed) in header_group if section[header]])

                doc = Document(page_content=course_content,
                               metadata={"source_url": course_url,
                                         "department": section["department"],
                                         "course_number": section["number"],
                                         "year": section["year"],
                                         "semester": section["semester"]})

                result_list.append(doc)
                courses_processed.add(course_name)

        # add info about this *section* of the course
        instructors = None
        if "instructors" in section and section["instructors"] is not None:
            # only include the instructor names, we don't care about IDs
            instructors = [instructor["name"]
                           for instructor in section["instructors"]]

        section_content = f"{section_name} "
        section_content += ", ".join(
            [f"{segment_parsed}: {section[segment]}" for (segment, segment_parsed) in section_headers] + [f"Instructors: {instructors}"])

        doc = Document(page_content=section_content,
                       metadata={"source_url": course_url,
                                 "department": section["department"],
                                 "course_number": section["number"],
                                 "year": section["year"],
                                 "semester": section["semester"]})
        result_list.append(doc)


##########################################
# Add passages from the CS major requirements and
# department info PDF. This was manually extracted
# and cleaned up from the PDF. Cite the PDF as the
# source for these passages.
##########################################
cs_major_documents = []
with open(f"{DEV_SET_PATH}/cs_pdf_gold.txt", "r") as f:
    for line in f:
        cs_major_documents.append(Document(
            page_content=line,
            metadata={"source_url": "https://catalog.williams.edu/pdf/csci.pdf",
                      "department": "CSCI"}))


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
def write_document_list_to_json(outpath, doc_list):
    with open(outpath, "w") as f:
        out = json.dumps([{"page_content": doc.page_content,
                           "metadata": doc.metadata} for doc in doc_list])
        f.write(out)


write_document_list_to_json(
    f"{DEV_SET_PATH}/cs_courses_documents.json", cs_courses_documents)
write_document_list_to_json(
    f"{DEV_SET_PATH}/cs_major_documents.json", cs_major_documents)


##########################################
# Create vectorstores from the data, then
# persist the vectorstores to disk.
##########################################

# 🔴 Uncomment the lines below to write the new vectorestore db   🔴
# 🔴 This may cause the model's behavior to change since it will  🔴
# 🔴 create a new knowledge store. It also costs API credits.     🔴

embeddings = OpenAIEmbeddings()

cs_major_vectorstore = Chroma.from_documents(
    cs_major_documents,
    embeddings,
    persist_directory=f"{DEV_SET_PATH}/cs_major_vectorstore")
# persist the db
cs_major_vectorstore.persist()


cs_courses_vectorstore = Chroma.from_documents(
    cs_courses_documents,
    embeddings,
    persist_directory=f"{DEV_SET_PATH}/cs_courses_vectorstore")
# persist the db
cs_courses_vectorstore.persist()
