from model_pipeline import get_model_pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, load_tools, initialize_agent
import pandas as pd

#model_id = "tiiuae/falcon-40b"
#model_name = "falcon-40b-instruct"
#model_name = "falcon-40b"
#model_name = "open_llama13b"
#model_name = "xgen-7b-8k-base"
model_name = "llama-65b"

pipe = get_model_pipeline(model_name)
llm = HuggingFacePipeline(model_id = model_name, pipeline = pipe)


df = pd.read_csv("/data/preprocessed/logs/2023-06-28/error.csv")
tools = load_tools(["python_repl"])
agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
own_description = "I have given you access to a dataframe called df, the columns are: {} you can execute pandas queries by using the python tool. Give me the name of the host that wrote the most amount of errors?".format(df.columns.tolist())
agent.run(own_description)

