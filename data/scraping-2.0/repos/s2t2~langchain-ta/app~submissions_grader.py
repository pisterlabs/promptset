
import os
from functools import cached_property
from pprint import pprint

from pandas import DataFrame
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain

from app import RESULTS_DIRPATH
from app.submissions_processor import SubmissionsProcessor
from app.document_processor import CHUNK_OVERLAP, CHUNK_SIZE, SIMILARITY_THRESHOLD
from app.rows_processor import RowsDocumentProcessor
from app.prompts import STUDENT_QUERY
from app.prompts.homework_4 import HOMEWORK_QUESTIONS
from app.submissions_retriever import SubmissionsRetriever, UNIQUE_ONLY, DOCS_LIMIT
from app.openai_llm import create_llm, MODEL_NAME, TEMP
from app.response_formatters import Student, QuestionScoring, ZERO_TO_ONE_SCORE, COMMENTS, CONFIDENCE_SCORE


def get_relevant_docs(retriever, query, verbose=True):
    relevant_docs_with_state = retriever.get_relevant_documents(query)
    #relevant_docs_with_state[0] #> _DocumentWithState has page_content and state["embedded_doc"]
    if verbose:
        print(query)
        print(len(relevant_docs_with_state))

    relevant_docs = [dws.to_document() for dws in relevant_docs_with_state]
    return relevant_docs


  
#QA_CONTEXT_TEMPLATE = """Answer the **query**, based on the provided **context**.
#
#**Context**: {context}
#
#**Query**: {query}
#"""

QA_CONTEXT_TEMPLATE = """Answer the **query**, based only on the provided **context**, and format your response according to the **formatting instructions** (avoid using special characters).

**Query**: {query}

**Context**: {context}

**Formatting Instructions**: {formatting_instructions}
"""

def qa_chain(llm, query, compression_retriever, parser_class, verbose=False):
    # https://www.youtube.com/watch?v=yriZBFKE9JU

    prompt = ChatPromptTemplate.from_template(QA_CONTEXT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

    relevant_docs = get_relevant_docs(retriever=compression_retriever, query=query)

    parser = PydanticOutputParser(pydantic_object=parser_class)
    formatting_instructions = parser.get_format_instructions()

    response = chain.invoke({"query": query, "context": relevant_docs, "formatting_instructions": formatting_instructions})
    parsed_response = parser.invoke(response["text"])
    return parsed_response


#QUESTION_SCORING_INSTRUCTIONS = f"""
#You are a helpful and experienced machine learning practitioner and instructor (i.e. the "grading assistant").
#Your goal is to accurately grade a student's machine learning homework assignment.
#You will be provided a question, and your task is to provide a score and corresponding comment,
#based on some provided context about the student's response.
#
#  + What 'score' would you give the response for this question? {ZERO_TO_ONE_SCORE}
#
#  + And why (i.e. your 'comments')? {COMMENTS}
#
#  + And how sure are you about this score (i.e. your 'confidence'), as a percentage between 0 (low confidence) and 1 (high confidence)? {CONFIDENCE_SCORE}
#
#NOTE: It is important to grade accurately and fairly, so if you don't know, we'd rather you provide a low confidence, and a low score, and a comment saying you're not sure.
#
#NOTE: If you don't have any context, or if you don't think the context is relevant enough, you can provide a zero.
#"""


QUESTION_SCORING_INSTRUCTIONS = f"""
You are an experienced machine learning practitioner and instructor (i.e. the Grader).
Your goal is to accurately grade a student's machine learning homework assignment.
You will be provided a question that the student was supposed to answer,
and your task is to grade how well the student answered that question,
based only on some context provided about the student's response.

Grading Guidance:

  + What 'score' would you give the response for this question?
    If you don't have any context, or if you don't think the context is relevant enough, you should assign a score of 0.
    If the student's response was off-topic, not specific enough, or not what the question is looking for, you should assign a score of 0.5.
    If the response was generally good, but there were some minor issue(s), you should assign a score of 0.75.
    If the response was relevant and correct, you should assign a score of 1.0.
    If the response was relevant and correct, and very thorough and detailed, you should assign a score of 1.25.

  + How certain are you about this score (i.e. your 'confidence')?

  + And why (i.e. your 'comments' about the score and/or the confidence)?
    You should provide specific justification for the score.
    You should cite specific content present or absent from the response, as well as your reasoning for providing the score.

REMEMBER: It is very important to grade accurately, so it is imperative that you only grade based on the provided context,
and you will prefer to give low confidence and a corresponding comment if you're not sure or if you don't have the context you need.
"""


class SubmissionsGrader(SubmissionsRetriever):

    def __init__(self, unique_only=UNIQUE_ONLY, similarity_threshold=SIMILARITY_THRESHOLD, docs_limit=DOCS_LIMIT,
                 #retrieval_strategy="chunks",
                 chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                 homework_questions=HOMEWORK_QUESTIONS,
                 results_dirpath=RESULTS_DIRPATH,

                 model_name=MODEL_NAME, temp=TEMP
                 ):

        super().__init__(unique_only=unique_only, similarity_threshold=similarity_threshold, docs_limit=docs_limit,
                 #retrieval_strategy="chunks",
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                 homework_questions=homework_questions, results_dirpath=results_dirpath
                )

        # ADDITIONS:
        self.temp = temp
        self.model_name = model_name
        self.llm = create_llm(model_name=self.model_name, temp=self.temp)

        self.scorings_csv_filepath = os.path.join(self.results_dirpath, f"scorings_similarity_{self.similarity_threshold}_chunks_{self.chunk_size}_{self.chunk_overlap}_temp_{self.temp}.csv")
        self.scorings_df = DataFrame()
        #self.errors_csv_filepath = os.path.join(self.results_dirpath, f"scoring_errors_similarity_{self.similarity_threshold}_chunks_{self.chunk_size}_{self.chunk_overlap}_temp_{self.temp}.csv")
        #self.errors_df = DataFrame()


    def perform(self):
        sp = SubmissionsProcessor()
        sp.perform()

        cells_df = sp.cells_df.copy()
        print("ALL CELLS:", len(cells_df))
        if self.unique_only:
            cells_df = cells_df[ cells_df["dup_content"] == False ]
            print("UNIQUE CELLS:", len(cells_df))

        submission_filenames = cells_df["filename"].unique()
        print("SUBMISSIONS:", len(submission_filenames))

        records = []
        errors = []
        submission_filenames = submission_filenames[0:self.docs_limit] if self.docs_limit else submission_filenames
        for filename in submission_filenames:
            print("---------------------")
            print(filename)
            rows_df = cells_df[ cells_df["filename"] == filename ]
            dp = RowsDocumentProcessor(rows_df=rows_df, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) # similarity_threshold=similarity_threshold
            #text_retriever = dp.text_compression_retriever
            base_retriever = dp.make_retriever(cell_type="TEXT", storage_strategy=self.retrieval_strategy)
            compression_retriever = dp.make_compression_retriever(base_retriever=base_retriever, similarity_threshold=self.similarity_threshold)

            record = {"filename": filename, "file_id": dp.file_id} # flattened structure, one row per submission document
            try:

                student = qa_chain(llm=self.llm, query=STUDENT_QUERY, compression_retriever=compression_retriever, parser_class=Student)
                record = {**record, **{"student_id": student.net_id, "student_name": student.name}}

                i = 1
                for query_id, query in self.homework_questions:
                    query = f"{QUESTION_SCORING_INSTRUCTIONS} {query}"
                    scoring = qa_chain(llm=self.llm, query=query, compression_retriever=compression_retriever, parser_class=QuestionScoring)
                    record[f"scoring_{i}_question_id"] = scoring.question_id
                    record[f"scoring_{i}_score"] = scoring.score
                    record[f"scoring_{i}_comments"] = scoring.comments
                    record[f"scoring_{i}_confidence"] = scoring.confidence
                    i+=1

                record["error"] = None
                records.append(record)
            except Exception as err:
                print("ERROR...")
                errors.append({"filename": filename, "error": err})
                records.append({"filename": filename, "error": err})

            print("-----------------")
            print("RECORDS:")
            print(len(records))
            print("-----------------")
            print("ERRORS:")
            pprint(errors)

        self.scorings_df = DataFrame(records)
        self.scorings_df.to_csv(self.scorings_csv_filepath, index=False)

        #self.errors_df = DataFrame(errors)
        #self.errors_df.to_csv(self.errors_csv_filepath, index=False)


        
if __name__ == "__main__":

    grader = SubmissionsGrader()
    grader.perform()

    print(grader.scorings_df.head())
