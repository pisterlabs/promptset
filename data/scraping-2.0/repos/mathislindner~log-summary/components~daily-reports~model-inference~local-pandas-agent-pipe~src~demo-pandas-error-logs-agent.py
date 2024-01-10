from model_pipeline import get_model_pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd

#model_id = "tiiuae/falcon-40b"
#model_name = "falcon-40b-instruct"
model_name = "falcon-40b"
#model_name = "xgen-7b-8k-base"

pipe = get_model_pipeline(model_name)
llm = HuggingFacePipeline(model_id = model_name, pipeline = pipe)



class MyAgent:
    #llm is a function that takes a prompt and returns a string
    def __init__(self, llm):
        self.llm = llm

    def start(self, question):
        #TODO import necessary dataframe here (maybe default to last 24hours)
        df_errors = pd.read_csv("/data/preprocessed/logs/2023-06-28/error.csv")
        #get pandas query from first chain execution
        pandas_query = self.first_chain(df_errors, question)
        #execute query
        first_chain_answer = df_errors.query(df_errors, pandas_query)
        # check on what the answer is (we need to do this because if it is a full dataframe, we cannot just put that into the llm)
        if first_chain_answer is str or first_chain_answer is int:
            #if it is a string or int, we can just put it into the llm and ask for a description
            pass
    def first_chain(df_errors, question):
    #first chain to translate english to a pandas query
        pandas_template = PromptTemplate(
            input_variables=["df_name","df_columns","question"],
            template="""
            INSTRUCTIONS:
            You are an expert at converting questions into pandas queries.
            You need to find a pandas query for a dataframe called {df_name}.
            The columns of the dataframe are: {df_columns}.
            Your goal is to create a pandas query for the dataframe to answer a Question. 
            The question you have to convert comes next.
            QUESTION: {question}
            The conversion of the question comes next.
            PANDAS QUERY: """
            )
        chain_get_pandas_query = LLMChain(llm = llm, prompt = pandas_template, verbose = True)
        answer = chain_get_pandas_query.run({'df_name': 'df_errors','df_columns': str(df_errors.columns.tolist()), 'question': 'which host wrote the most amount of errors?'})
        print('answer:', answer)
        #only use what is between the answer tags <ANSWER></ANSWER>
        #select first sentence
        first_line = answer.splitlines()[1]
        return first_line

    def short_answer(df_errors, question, answer):
        pass

    def long_answer(df_errors, question, answer):
        pass


    def query(df_errors, pandas_query):
        #TODO:this is very unsafe, easy to inject code
        result = None
        exec('result = ' + pandas_query)
        return result
    
    ############################################################################################################

    #second chain to explain the queries answer
    #answer can be an integer, string or pandas dataframe
    if type(result) is int or type(result) is str:
        str_int_template = PromptTemplate(
            input_variables=["answer","question"],
            template="INSTRUCTIONS You are an friendly assistant that works in a sysadmin department. You will be given a QUESTION where the ANSWER has already been found. Your goal to retranscribe the blunt answer in a more eloquent way. Delimit the answer with a tag 'MESSAGE:'. \nHere is the question: \nQUESTION: {question}\n This is the answer to it\nANSWER: {answer}\n",
            )
        chain_explain_answer = LLMChain(llm = llm, prompt = str_int_template, verbose = True)
        message = chain_explain_answer.run({"answer": str(result), "question": "how many errors are there?"})

    elif type(result) is pd.DataFrame:
        #if the dataframe is small enoguh
        print("answer:", answer)