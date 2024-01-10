from langchain.llms import GooglePalm
from langchain import PromptTemplate, LLMChain
from fastapi import  HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain import PromptTemplate, LLMChain
from fastapi.responses import FileResponse


def string_to_html(input_string, output_file):
    # Create and open the HTML file
    with open(output_file, 'w') as html_file:
        # Write the HTML content
        html_file.write(input_string)


llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=1024,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

db = SQLDatabase.from_uri("sqlite:///./data_KRA.sqlite")
toolkit = SQLDatabaseToolkit(db=db , llm=llm)

PROMPT = '''You are an agent designed to interact with a SQL database.

Here are the relations between tables:

1. RM_KRAs (EMPLOYEE_ID) conected to EMPLOYEES (EMP_ID)
2. EMPLOYEES (EMP_ID) conected to CUSTOMERS_EMPLOYEES (RM)
3. EMPLOYEES (EMP_ID) conected to CONTACTHISTORY (RM)
4. CUSTOMERS_EMPLOYEES (CUST_ID) conected to CUSTOMERS (CUST_ID)
5. CUSTOMERS_EMPLOYEES (CUST_ID) conected to PERSONA (CUST_ID)
7. CUSTOMERS_EMPLOYEES (CUST_ID) conected to PRODUCT_HOLDING (CUST_ID)

Here are the explanations of what the tables mean:

1. RM_KRAs: This contains RM level yearly targets for different KRAs for FY22-23 with achievements till Dec’22.
2. Employees: Employee Dimension table containing roll-up to regions 
3. Customers_Employee: Mapped book for each RM as of Dec’22
4. Customers: Customer Dimension table 
5. Persona: Customer Persona details
6. Product_holdings: Customer product holding as of end of Dec’22
7. Contacthistory: Customers contacted by different RMs from Jul-Dec’22 with disposition

Columns of all the tables are listed below:

1. These are the columns of RM_KRAs table:
[
    'Employee_ID', 'TARGET', 'Unit', 'Target_FY22_23_ABS', 'Target_FY22_23_PCT',
    'Rating', 'CURR_COMPLETION_ABS', 'CURR_COMPLETION_PCT', 'APR_COMPLETION_ABS',
    'APR_COMPLETION_PCT', 'MAY_COMPLETION_ABS', 'MAY_COMPLETION_PCT', 'JUN_COMPLETION_ABS',
    'JUN_COMPLETION_PCT', 'JUL_COMPLETION_ABS', 'JUL_COMPLETION_PCT', 'AUG_COMPLETION_ABS',
    'AUG_COMPLETION_PCT', 'SEP_COMPLETION_ABS', 'SEP_COMPLETION_PCT', 'OCT_COMPLETION_ABS',
    'OCT_COMPLETION_PCT', 'NOV_COMPLETION_ABS', 'NOV_COMPLETION_PCT', 'DEC_COMPLETION_ABS',
    'DEC_COMPLETION_PCT', 'JAN_COMPLETION_ABS', 'JAN_COMPLETION_PCT', 'FEB_COMPLETION_ABS',
    'FEB_COMPLETION_PCT', 'MAR_COMPLETION_ABS', 'MAR_COMPLETION_PCT'
]

2. These are the columns of Employees table:
[
    'Emp_ID', 'Name', 'Email', 'SOL_ID', 'Cluster', 'Circle', 'Region', 'Branch_Type'
]

3. These are the columns of Customers_Employee table:
[
    'Cust_ID', 'RM', 'ACCT_BAL', 'ACCT_BAL_FY_START'
]

4. These are the columns of Customers table:
[
    'Cust_ID', 'Name', 'Age', 'Gender', 'Location', 'Marital_Status', 
    'Education', 'Occupation', 'MOB', 'Income', 'Dependents', 
    'Digital_ind', 'Email', 'Phone', 'Address'
]

5. These are the columns of Persona table:
[
    'Cust_ID', 'Location_Type', 'Investment_risk_tol', 'Avg_mon_expense', 
    'Investment_needs', 'BAnking_Needs', 'Pref_channel', 'Lifestyle', 
    'Net_Worth', 'Persona', 'Biz_Type', 'Biz_Size', 'Biz_Age', 'Turnover', 
    'Credit_Score'
]

6. These are the columns of Product_holdings table:
[
    'Cust_ID', 'Term_Deposit', 'Auto_Loan', 'Two_Wheeler_Loan', 'Personal_Loan',
    'Home_Loan', 'Credit_Card', 'Life_Insurance', 'Mutual_Fund', 'General_Insurance',
    'Agri_Loan', 'National_Pension_Scheme', 'Tractor_Loan', 'Remittance', 'Forex_Card',
    'Trading_Account', 'Digital_Banking', 'Credit_Card_CLI', 'Credit_Card_EMI',
    'Credit_Card_Upgrade', 'Education_Loan'
]

7. These are the columns of Contacthistory table:
[
    'Cust_ID', 'RM', 'contact_date', 'Product', 'disposition'
]


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

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix = PROMPT,
    suffix = Suffix
    
)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query_tables/")
def query_sql(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")
    result = agent_executor.run(query['query'])
    return  result

@app.post("/query_reports/")
def query_sql(query: dict):
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query missing in request")
    result = agent_executor.run(query['query'] + "limit the data points to 10")
    template = """Hello AI Bot,
    You're an expert in text data analysis, 
    specializing in creating detailed HTML pages with three 
    types of graphical visualizations (e.g., Bar, Pie, Histogram) 
    and descriptive analysis. You transform provided data into 
    comprehensive HTML pages featuring graphical representations and insights.

    content: {content}

    HTML Page: Make sure that the complete HTML code is given
    Example Report:
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Metrics</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>

    <h1>Performance Metrics for Aarav Employee (ETB_MDAB KRA)</h1>

    <div>
        <p>Target: 3565.223244 (in lakhs)</p>
        <p>Target_FY22_23_ABS: 100</p>
        <p>Target_FY22_23_PCT: 2</p>
        <p>Rating: 2</p>
    </div>

    <div id="performance-chart"></div>

    <script>
        // Data from provided metrics
        var months = ['APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR'];
        var completion_abs = [404.7562297, 355.2744963, 336.617683, 386.3596777, 413.7227661, 212.1058265, 312.9125137, 306.4131117, 402.5065738, 0, 0, 0];
        var completion_pct = [11.3529, 9.965, 9.4417, 10.8369, 11.6044, 5.9493, 8.7768, 8.5945, 11.2898, 0, 0, 0];

        var trace1 = {{
            x: months,
            y: completion_abs,
            type: 'bar',
            name: 'Completion (Absolute)',
            yaxis: 'y'
        }};

        var trace2 = {{
            x: months,
            y: completion_pct,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Completion (Percentage)',
            yaxis: 'y2'
        }};

        var data = [trace1, trace2];

        var layout = {{
            title: 'Monthly Completion Metrics',
            xaxis: {{title: 'Month'}},
            yaxis: {{
                title: 'Completion (Absolute)',
                overlaying: 'y2'
            }},
            yaxis2: {{
                title: 'Completion (Percentage)',
                side: 'right',
                overlaying: 'y'
            }}
        }};

        Plotly.newPlot('performance-chart', data, layout);
    </script>

    </body>
    </html>
    """
    prompt = PromptTemplate(template=template, input_variables=["content"])

    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm
    )

    question = result
    output = llm_chain.run(question)
    stringContent = output.replace("```html", "").replace("```", "")
    string_to_html(stringContent, "index.html")

    return FileResponse("index.html", media_type="text/html")
    

