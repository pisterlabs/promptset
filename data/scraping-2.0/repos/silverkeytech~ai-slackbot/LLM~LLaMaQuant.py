from LLM.LLmMain import LLM

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

class LLaMaQuant(LLM):
    modelPath = "/home/ahmedaymansaad/Code/LLModels/llama-2-7b-chat.ggufv3.q4_K_M.bin"
    nGpuLayers = 1
    nBatch = 512
    nCtx = 4096
    nThreads = 2
    callbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

    def __init__(self):
        self.InitializeModel()
        testStatus = self.quickTestModel()
        if testStatus == False:
            raise Exception("Model failed to pass quick test")


    def InitializeModel(self):
        self.llm = LlamaCpp(
            model_path=self.modelPath,
            callback_manager=self.callbackManager,
            verbose=True,
            n_ctx=self.nCtx,
            n_batch= self.nBatch,
            n_gpu_layers=self.nGpuLayers,
            f16_kv=True,
            streaming = True,
            seed = 0,
        )

    def quickTestModel(self):
        output = self.llm("Q: What is the capital of France? A:", stop=["Q:", "\n"],echo=False)
        if output.lower().find("paris") != -1:
            print("Test passed")
            return True
        else:
            print("Test failed")
            print(output)
            return False
        
    def agent(self, question):
        template = """[INST]
        <<SYS>>
        You are an internal choice agent, that takes in a question and decides which tool to use to answer it.
        The tools that are provided are:
        - GithubLogs:   This contains github disscussions where users write their daily logs and progress.
                    Use this tool to answer questions about what users have done in their day.
        - Attendance:   This contains attendance information (working/finished/away).
                        Use this tool to answer questions about the duration and work hours.
        - Users:    This contains information about the users such as names, emails, usernames.
                    Use this tool to answer questions about the users and their information
        - DailyReport,n:    This contains the markdown for the daily github discussion.
                            n is the number of days ago, 0 is today, 1 is yesterday, etc.
                            Use this tool to when asked to give the daily report.

        You must always give your answer in one word which is the name of the tool that is best suited to answer the question.
        Therefore you must always return one of the following words:
        - GithubLogs
        - Attendance
        - Users
        - DailyReport

        Your can only answer using words from the previous list.

        For example, if the question is "What did Ahmed do today?", the answer is "GithubLogs".
        If the question is "When did Ahmed start working?", the answer is "Attendance".
        If the question is "What is Ahmed's email?", the answer is "Users".
        If the question is "How long did Ahmed work for on 10/7?", the answer is "Attendance".
        If the question is "Give me the daily report.", the answer is "DailyReport,0".
        If the question is "Give me the daily report from 2 days ago.", the answer is "DailyReport,2".
        If the question is "Give me yesterday's daily report.", the answer is "DailyReport,1".


        <</SYS>>
        Question: {question}
        [/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        output = llm_chain({'question':question})
        print(output)
        return output["text"]
        
    def respond(self, question, context, contextType):
        template = """
        <<SYS>>
        You are Sil, a friendly chatbot assistant that responds conversationally to users' questions.
        Keep the answers short, Lower character count answers are preferred.
        If you do not know the answer to a question say that you do not know, don't try to guess.
        You have context that you can use to answer questions, if the context is not relevant to the question, you can ignore it.
        All information in the context is public information, you can use it to answer questions.
        Types of context:
        - Github: This contains github disscussions where users write their daily logs and progress.
        - Attendance: This contains attendance information for the users (when they started working/finished/away).
        - Users: This contains information about the users (name, email, github username, slack username, etc).
        - DailyReport: This contains the markdown for the daily github discussion.
                        Report requirments:
                        If you are asked to give the daily report, you must try to summarize what every user did and return it in presentable report like manner.
                        First row contains the heading the of the discussion and the rest of the rows contain the users' logs.
                        User name are in the authors column.
                        Place your summary report inside a markdown block/blob.
                        Remove any unnecessary white space.

        Context Type: 

        {contextType}

        Context: 
        
        {context}

        <</SYS>>
        [INST]

        Question: {question}

        [/INST]"""
        prompt = PromptTemplate(template=template, input_variables=["question", "context", "contextType"])

        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        output = llm_chain({'question':question,'context':context, 'contextType':contextType})
        print(output)
        return output["text"]
        