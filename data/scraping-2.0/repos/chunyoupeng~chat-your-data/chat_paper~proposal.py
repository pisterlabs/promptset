from query_data import *
from langchain.prompts.prompt import PromptTemplate


class Proposal:
    def __init__(self, catalog_path, question_lst) -> None:
        self.catalog_path = catalog_path
        self.question_lst = question_lst
        self.proposal_path = self.catalog_path.replace("catalog", "proposal")
        self.template = """论文大纲:{catalog}
        请你根据上面的论文名和论文大纲,帮我写一个关于{question}相关的内容,写一个段落,作为我论文开题报告的一部分.
        输出:
        """
        # self.prompt = PromptTemplate.from_template(template=template)
        
    def __call__(self):
        pompt = PromptTemplate.from_template(template=self.template)
        llm = get_llm(name='openai', temperature=0.3)
        chain = pompt | llm | StrOutputParser()
        self.write_file(chain)

    def write_file(self, chain):
        with open(self.catalog_path, "r") as f:
            catalog_content  = f.read()
        with open(self.proposal_path, "w", encoding="utf-8") as f:
            for q in self.question_lst:
                print(q)
                rt = chain.invoke({"catalog": catalog_content, "question": q})
                f.write(f"{q}\n{rt}\n\n")
    
def main():
    catalog_path = sys.argv[1]
    question_path = sys.argv[2]
    with open(question_path, "r") as f:
        question_lst = f.read().split("\n")
    proposal = Proposal(catalog_path, question_lst)
    proposal()

if __name__ == '__main__':
    main()