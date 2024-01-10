import os

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import E2BDataAnalysisTool

os.environ["E2B_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Artifacts are charts created by matplotlib when `plt.show()` is called
def save_artifact(artifact):
    print("New matplotlib chart generated:", artifact.name)
    # Download the artifact as `bytes` and leave it up to the user to display them (on frontend, for example)
    file = artifact.download()
    basename = os.path.basename(artifact.name)

    # Save the chart to the `charts` directory
    with open(f"./charts/{basename}", "wb") as f:
        f.write(file)



def process_csv_with_langchain(file, question, description):
    e2b_data_analysis_tool = E2BDataAnalysisTool(
        # Pass environment variables to the sandbox
        env_vars={"MY_SECRET": ""},
        on_stdout=lambda stdout: print("stdout:", stdout),
        on_stderr=lambda stderr: print("stderr:", stderr),
        on_artifact=save_artifact,
    )

    with open(file, 'rb') as file_object:  # Ensure consistent variable naming
        remote_path = e2b_data_analysis_tool.upload_file(file=file_object, description=description)
        print(remote_path)

    tools = [e2b_data_analysis_tool.as_tool()]

    template = """You are a professional farmer and analyst always calculate these depending on what data:
       Feed Conversion Ratio (FCR) = Total Feed Intake / Weight Gain
       Mortality Rate = (Number of Deaths / Total Number of Animals) x 100%
       Average Daily Gain (ADG) = Total Weight Gain / Number of Days
       Crop Yield = Total Production (units) / Cultivated Area (acres or hectares)
       Profit per Acre = (Yield per Acre x Price per Unit) - Cost per Acre
       Break-Even Point = Fixed Costs / (Price per Unit - Variable Cost per Unit)
       Repayment Capacity = Net Farm Income + Off-Farm Income - Family Living Expenses
       Debt-to-Asset Ratio = Total Debt / Total Assets
       Operating Expense Ratio = Total Operating Expenses / Gross Farm Income
       Current Ratio = Current Assets / Current Liabilities
       Quick Ratio = (Current Assets - Inventory) / Current Liabilities
       ROA = Net Farm Income / Average Total Farm Assets
       ROI = (Net Return on Investment / Cost of Investment) x 100%
        """

    template2 = "You are a professional farmer and analyst always calculate these depending on what data"
    

    llm = ChatOpenAI(model="gpt-4", temperature=0 )
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
    )

    
    ai_response = agent.run(question)

    # Close the sandbox
    e2b_data_analysis_tool.close()

    return ai_response