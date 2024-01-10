from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from library.chains import create_sql_query_chain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from config.config_vectordb import VectorDB
load_dotenv('.env')

class reportController:
    def __init__(self) -> None:
        self.vector_db = VectorDB()

    def excute_query(self,request):
        requestJson = request.get_json()
        question = requestJson["question"]
        prompt = self.get_training_sql_prompt(question)
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)
        chain = create_sql_query_chain(llm=llm,question=question)
        query = chain.invoke({"question": prompt.format(input=question)})
        answer = query.replace("\n", " ")
        result = {'question': question, 'answer': answer}
        return result

    def get_training_ddl_prompt(self,question):
        docs = self.vector_db.connect_vectordb('training_ddl').similarity_search(query=question,k=5)
        tables = "Only use information from the ddl below to write queries: \n"
        for value in docs:
            tables += value.page_content+'-'+value.metadata['table']+'\n'
        return tables

    def get_training_sql_prompt(self,question):
        docs = self.vector_db.connect_vectordb('training_sql').similarity_search(query=question,k=3)
        questions = []
        for value in docs:
            question = {}
            question['question'] = value.page_content
            question['answer'] = value.metadata['query']
            questions.append(question)
        example_prompt = PromptTemplate(input_variables=["question", "answer"], 
                                        template=
                                        """
                                            When query compare name add N'' 
                                            'Danh sách' and 'Báo cáo' and 'Thống kê' have the same meaning
                                            Some examples of SQL queries that correspond to questions are:
                                            {question}\n answer:{answer}
                                        """
                                        )
        prompt = FewShotPromptTemplate(
            examples=questions,
            example_prompt=example_prompt,
            suffix="Question: {input}",
            input_variables=["input"]
        )
        return prompt