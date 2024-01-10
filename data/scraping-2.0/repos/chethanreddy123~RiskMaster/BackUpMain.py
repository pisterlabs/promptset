from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from langchain.utilities import SQLDatabase
from fastapi.responses import JSONResponse
from langchain import PromptTemplate, LLMChain
from fastapi import  HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder 
from langchain import PromptTemplate
from pymongo import MongoClient
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from starlette.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, Form, UploadFile
import PIL.Image
import io
import google.generativeai as genai
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


GOOGLE_API_KEY = 'AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
genai.configure(api_key=GOOGLE_API_KEY)


def extract_text_from_first_page(pdf_file):
    # Open the PDF
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    page = doc[0]  # Extract text from the first page

    # If the PDF contains text directly
    if page.get_text():
        return page.get_text()

    # If the PDF contains images, use OCR
    text = ""
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        text += pytesseract.image_to_string(image)

    return text

def predict_with_models(sample_input):

    model = joblib.load('XGBClassifier_model.joblib')
    
    data = pd.DataFrame([sample_input])
    print(sample_input)
    
    le = LabelEncoder()
    obj_col = ['HasCoSigner','LoanPurpose','HasDependents', 'HasMortgage','MaritalStatus', 'EmploymentType', 'Education']
    for col in obj_col:
        data[col] = le.fit_transform(data[col])

    data = data.drop(['LoanID'], axis=1)
    
    trained_models = joblib.load('XGBClassifier_model.joblib')
    y_pred = model.predict(data)
    return y_pred

api_key = "AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM"
llm = GoogleGenerativeAI(model="models/gemini-pro", google_api_key=api_key)

model_image = genai.GenerativeModel('gemini-pro-vision')

MongoCli = MongoClient("mongodb+srv://chethan1234:1234@cluster0.uou14qn.mongodb.net/?retryWrites=true&w=majority")
LoanData = MongoCli["RiskMaster"]["LoanData"]


db = SQLDatabase.from_uri("sqlite:///./Loan_default.db")
toolkit = SQLDatabaseToolkit(db=db , llm=llm)




PROMPT = '''You are an agent designed to interact with a SQL database.

The dataset comprises information related to loan applicants, encompassing various demographic and financial attributes. Here is a detailed description of each field in the dataset:

1. LoanID: A unique identifier assigned to each loan application.

2. Age: The age of the loan applicant, indicating the number of years since birth.

3. Income: The total income of the applicant, representing their financial earnings.

4. LoanAmount: The amount of money requested by the applicant as a loan.

5. CreditScore: A numerical representation of the creditworthiness of the applicant, often used by lenders to assess the risk of default.

6. MonthsEmployed: The duration, in months, for which the applicant has been employed.

7. NumCreditLines: The number of credit lines the applicant currently holds.

8. InterestRate: The rate at which interest is charged on the loan amount.

9. LoanTerm: The duration of the loan in months, indicating the period within which the loan must be repaid.

10. DTIRatio (Debt-to-Income Ratio): The ratio of the applicant's total debt to their total income, providing insight into their financial obligations.

11. Education: The educational qualification of the applicant (e.g., Bachelor's, Master's, High School).

12. EmploymentType: The type of employment status of the applicant (e.g., Full-time, Unemployed).

13. MaritalStatus: The marital status of the applicant (e.g., Married, Divorced, Single).

14. HasMortgage: Indicates whether the applicant has an existing mortgage (Yes/No).

15. HasDependents: Indicates whether the applicant has dependents (Yes/No).

16. LoanPurpose: The purpose for which the loan is requested, providing context to the intended use of funds.

17. HasCoSigner: Indicates whether there is a co-signer for the loan (Yes/No).

18. Default: A binary indicator (0 or 1) representing whether the applicant has defaulted on a loan (1) or not (0).

This dataset is valuable for building predictive models to assess the risk of loan default based on various applicant characteristics. The goal is to leverage this information to create a technology solution, like RiskMaster, that utilizes Large Language Models for accurate default prediction and prevention in the Indian banking sector.


Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
If you get a "no such table" error, rewrite your query by using the table in quotes.
DO NOT use a column name that does not exist in the table.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite a different query and try again.
DO NOT try to execute the query more than three times.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
If you cannot find a way to answer the question, just return the best answer you can find after trying at least three times.
'''

Suffix = '''Begin!
Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}'''

template = """Task: Perform an in-depth analysis as an expert banker and loan default predictor based on the user data provided. Generate a detailed report assessing the applicant's ability to repay the loan, estimating the number of installments they might skip, and other relevant insights.

UserData: {query}

User Attributes Description:

1. LoanID: A unique identifier assigned to each loan application.
2. Age: The age of the loan applicant, indicating the number of years since birth.
3. Income: The total income of the applicant, representing their financial earnings.
4. LoanAmount: The amount of money requested by the applicant as a loan.
5. CreditScore: A numerical representation of the creditworthiness of the applicant.
6. MonthsEmployed: The duration, in months, for which the applicant has been employed.
7. NumCreditLines: The number of credit lines the applicant currently holds.
8. InterestRate: The rate at which interest is charged on the loan amount.
9. LoanTerm: The duration of the loan in months, indicating the repayment period.
10. DTIRatio (Debt-to-Income Ratio): The ratio of the applicant's total debt to their total income.
11. Education: The educational qualification of the applicant (e.g., Bachelor's, Master's, High School).
12. EmploymentType: The type of employment status of the applicant (e.g., Full-time, Unemployed).
13. MaritalStatus: The marital status of the applicant (e.g., Married, Divorced, Single).
14. HasMortgage: Indicates whether the applicant has an existing mortgage (Yes/No).
15. HasDependents: Indicates whether the applicant has dependents (Yes/No).
16. LoanPurpose: The purpose for which the loan is requested, providing context to the intended use of funds.
17. HasCoSigner: Indicates whether there is a co-signer for the loan (Yes/No).

Analysis and Recommendations:
 """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix = PROMPT,
    suffix = Suffix
    
)


df = pd.read_csv("Loan_default.csv")

agent_pandas = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True,
    return_intermediate_steps=True)


def handle_query(agent, query):
    if "graph" in query.lower() or "plot" in query.lower():
        # Handle graph-related query
        # Assuming 'agent' can process the query and generate a graph
        graph_result = agent(query + " Note: save the plot as plot.png")
        return FileResponse("plot.png" , headers={"Content-Disposition": "attachment; filename=plot.png"})
    else:
        return str(agent(query))
    


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query_loans/")
def query_sql(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")
    result = agent_executor.run(query['query'])
    return  result

@app.post("/predict_loan/")
def predict_loan(user_input: dict):
    prediction = predict_with_models(user_input)
    if prediction[0] == 0:
        return "Loan will not default"
    else:
        return "Loan will default"
    
@app.post("/get_insigths_user/")
def get_insigths_user(user_input: dict):
        return (llm(
        prompt_template.format(
            query=user_input
        )
    ))
    

@app.post("/store_user_data/")
async def store_user_data(user_data: dict):
    try:
        # Insert user data into MongoDB
        result = LoanData.insert_one(user_data)

        # Return success message
        return {"message": "User data stored successfully!", "inserted_id": str(result.inserted_id)}

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error storing user data: {str(e)}")
    


@app.get("/top_ten_new_loans")
async def get_top_ten_new_loans():
    try:
        # Retrieve the top ten new loan data from MongoDB in reverse order
        top_ten_loans = list(LoanData.find().sort([("_id", -1)]).limit(10))

        for i in top_ten_loans:
            # Remove the MongoDB object ID from each loan data
            i.pop("_id")
        # Return the top ten new loan data
        return top_ten_loans

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving top ten new loans: {str(e)}")


@app.post("/get_user_data/")
async def get_user_data(user_data: dict):
    try:
        # Retrieve the user data from MongoDB
        user_data = LoanData.find_one(user_data)

        # Remove the MongoDB object ID from the user data
        user_data.pop("_id")

        # Return the user data
        return user_data

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving user data: {str(e)}")


@app.post("/get_pandas_agent_response/")
async def get_pandas_agent_response(query: dict):

    try:
        # Retrieve the response from the pandas agent
        response = handle_query(agent_pandas, query['query'])

        # Return the response
        return response

    except Exception as e:
        # Return error message if an exception occurs
        raise HTTPException(status_code=500, detail=f"Error retrieving pandas agent response: {str(e)}")
    


@app.post("/image_analysis/")
async def image_analysis(file: UploadFile = File(...), text: str = Form(...)):

    print("Image analysis request received")
    # Read image file
    image_content = await file.read()
    image = PIL.Image.open(io.BytesIO(image_content))

    # Process with Google Gemini Vision Pro

    response = model_image.generate_content([text, image], stream=True)
    response.resolve()

    return {
        "response" : str(response.text)
    }


@app.post("/pdf_loan_analysis/")
async def pdf_loan_analysis(file: UploadFile = File(...)):
    # Read PDF file
    contents = await file.read()

    # Extract text from PDF
    extracted_text = extract_text_from_first_page(contents)

    # Process with Google Gemini Pro
    response = llm(
        prompt_template.format(
            query="Given below is extracted text of document related for applying a loan get the relavent information so that it helps in the loan process:\n f{extracted_text}"
        )
    )

    
    important_info = "Processed text to extract loan-relevant information"

    return JSONResponse(content={"important_info": response})


