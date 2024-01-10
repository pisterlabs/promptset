
from langchain.prompts import PromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import openai
import re
import json
openai.api_key = "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"
OPENAI_API_KEY = "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"
# model_name='gpt-4'

model_name='gpt-3.5-turbo'
llm = ChatOpenAI(model_name=model_name,temperature=0,openai_api_key=OPENAI_API_KEY)
def clean_string(s):
    # Replace "A. ", "B. ", "C. ", "D. " and following whitespace with an empty string
    cleaned_s = re.sub(r'[ABCD]\.\s', '', s)
    return cleaned_s
def generateMCQS(text):
  prompt_templateMCQS = """You are a university professor making a quiz with your knowledge for your student which test conceptual and critical thinking skills
    of a student. I'll provide you with some information and
  ask you a MCQ question based on that source 
  and the correct answer.
  Your job is to explain why the correct option
  is correct and why other otions are incorrect(provide this information on the last of the response) on the source information I provide you also mention correct option at the end:

  {context}

  Question: {question}

    Please provide the evaluation in the form of a json. Following is the format of the json:
    [{{   
    "Question": "question",
    "Answer A": {{"Option A":"option A","Explanation A": "Short explanation of why the answer A is correct or incorrect"}},
    "Answer B": {{"Option B":"option B","Explanation B": "Short explanation of why the answer B is correct or incorrect"}},
    "Answer C": {{"Option C":"option C","Explanation C": "Short explanation of why the answer C is correct or incorrect"}},
    "Answer D": {{"Option D":"option D","Explanation D": "Short explanation of why the answer D is correct or incorrect"}},
    "Correct Option": "Correct option as, 'A', 'B', 'C', or 'D'"
    
    }}]

  Note: Don't give the options that are partially correct.

  I want to ensure that the format remains the same for each of the questions, options, correct answers, and explanations. Thank you.
  The questions should be open-ended, add examples if possible, focus on why more and at the high difficulty level for a university student.  

  """
  PROMPTMCQS = PromptTemplate(
      template=prompt_templateMCQS, input_variables=["context", "question"]
  )

  chainMCQS = LLMChain(llm=llm , prompt=PROMPTMCQS)
  questionMCQS = "Please generate two in-depth multiple-choice questions, each question should have four options. The questions should be about the main concepts, insights, and implications. Also mention in the explaination why the oprion is correct and why its not correct"
  ouptutMCQS = chainMCQS.predict(context=text, question=questionMCQS)
  # print(ouptutMCQS)
  return ouptutMCQS


def generateFAQs(text):
  prompt_templateFAQS = """You are a university professor making a quiz with your knowledge for your student which test conceptual and critical thinking skills
    of a student. I'll provide you with some information and ask you to generate open ended questions based on that source and the answer based on source.
    Question should be in the form that it requiress the Answer in two or three bullet points.

  Your job is to explain why the answer is correct(provide this information on the last of the response) based on the source information I provide you:

  {context}

  Question: {question}

  please generate the response in following format:

  1. Question?
  Answer: [Bullet point one] # [Bullet point two] # [Bullet point three]

  Explanation: [Explanation why the bullet one is correct] # [Explanation why the bullet two is correct] # [Explanation why the bullet three is correct]

  ####

    Please provide the evaluation in the form of a json. Following is the format of the json:
    [{{   
    "Question": "question",
    "Answer A": {{"Option A":"first answer","Explanation A": "why the first answer is correct or incorrect"}},
    "Answer B": {{"Option B":"second answer","Explanation B": "why the second answer is correct or incorrect"}},
    "Answer C": {{"Option C":"third answer","Explanation C": "why the third answer is correct or incorrect"}}
    }}]

  I want to ensure that the format remains the same for each of the questions, answers, and explanations. Thank you.
  The questions should be open-ended, add examples if possible, focus on why more and at the high difficulty level for a university student.  

  """
  PROMPTFAQS = PromptTemplate(
      template=prompt_templateFAQS, input_variables=["context", "question"]
  )
  chainFAQS = LLMChain(llm=llm , prompt=PROMPTFAQS)
  questionFAQS = "Please generate two in-depth FAQ questions. The questions should be about the main concepts, insights, and implications."
  ouptutFAQS = chainFAQS.predict(context=text, question=questionFAQS)
  return ouptutFAQS



def mcqGenerate(chunck):
    MCQ = generateMCQS(chunck)
    # print(MCQ)
    listQuestions = ''
    try:
      listQuestions = json.loads(MCQ)
    except:
       pass
    # print(listQuestions)
    listFinal = []
    for x in listQuestions:
        cleanExtracted = []
        questionDic = {}
        try:
            questionDic["question"] = x["Question"]
            questionDic["type"] = "MCQS"
            questionDic["choices"] = [clean_string(x["Answer A"]["Option A"]),clean_string(x["Answer B"]["Option B"]),clean_string(x["Answer C"]["Option C"]),clean_string(x["Answer D"]["Option D"])]
            questionDic["choicesDescriptions"] = [x["Answer A"]["Explanation A"],x["Answer B"]["Explanation B"],x["Answer C"]["Explanation C"],x["Answer D"]["Explanation D"]]
            questionDic["correctAnswer"] = x["Correct Option"]
            listFinal.append(questionDic)
        except:
            pass
    return listFinal
def faqGenerate(chunck):
    FAQ = generateFAQs(chunck)
    # print(FAQ)
    listQuestions = ''
    try:
      listQuestions = json.loads(FAQ)
    except:
       pass
    listFinal = []
    # print(listQuestions)
    for x in listQuestions:
        questionDic = {}
        try:
            questionDic["question"] = x["Question"]
            questionDic["type"] = "Descriptive"
            questionDic["topics"] = [clean_string(x["Answer A"]["Option A"]),clean_string(x["Answer B"]["Option B"]),clean_string(x["Answer C"]["Option C"])]
            questionDic["choicesDescriptions"] = [x["Answer A"]["Explanation A"],x["Answer B"]["Explanation B"],x["Answer C"]["Explanation C"]]
            listFinal.append(questionDic)
        except:
            pass
    # print(listFinal)
    return listFinal
def generateSummary(chunk):
  openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You need to summarize this text and don't mention its a summary."},
            {"role": "user", "content": chunk}
        ]
    )
  # print(openai_response['choices'][0]['message']['content'])
  return openai_response['choices'][0]['message']['content']



def gradeFAQs(dictQuestion):
  prompt_grading = """
    Context: {context}

    Question: {question}

    Student's Answers: 
    A. {student_answer1}
    B. {student_answer2}
    C. {student_answer3}

    You are a teacher grading a quiz based on given answers, and context. Evaluate the responses and grade them. The \
      grade should be out of 10. You need to give equal weightage to all of the answers \
      For every correct options give give 3 marks. For partially correct options give 1 \
      mark to each option. For each incorrect option give 0 marks. If all the options are correct give 10 marks. In the \
      end combine the grade of all the options and give a total grade for the question out of 10. Grade will be of the overall question. \
      Please provide the evaluation in the form of a json. Following is the format of the json:
    {{   
    "Question": "question",
    "Answer A": {{"Option A":"first answer","Explanation A": "why the first answer is correct or incorrect"}},
    "Answer B": {{"Option B":"second answer","Explanation B": "why the second answer is correct or incorrect"}},
    "Answer C": {{"Option C":"third answer","Explanation C": "why the third answer is correct or incorrect"}},
    "Grade": "total grade of the question"
    }}

    """
  PROMPTFAQS = PromptTemplate(
      template=prompt_grading, input_variables=["context", "question","student_answer1","student_answer2","student_answer3"]
  )
  chainFAQS = LLMChain(llm=llm , prompt=PROMPTFAQS)
  ouptutFAQS = chainFAQS.predict(context=dictQuestion["chunk"], question=dictQuestion["question"],student_answer1=dictQuestion["answers"][0],student_answer2=dictQuestion["answers"][1],student_answer3=dictQuestion["answers"][2])
  return ouptutFAQS

def faqGrade(dictQuestion):
    FAQ = gradeFAQs(dictQuestion)
    dict_obj = json.loads(FAQ)
    response_final = {"explanations":[dict_obj["Answer A"]["Explanation A"],dict_obj["Answer B"]["Explanation B"],dict_obj["Answer C"]["Explanation C"]], "grade": dict_obj["Grade"]}
    return response_final

