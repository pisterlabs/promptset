# from pygooglenews import GoogleNews
# from dotenv import load_dotenv
# import ast

# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain

# def harvey_helper(company):
#     # Required to load hidden variables from the environment
#     # In this case it is the API key for OpenAI
#     load_dotenv()

#     gn = GoogleNews()

#     res = gn.search(f"intitle:{company}")

#     structured = []

#     for i in res["entries"]:
#         structured.append(
#             {
#                 "title": i["title"],
#                 "link": i["link"],
#                 "date": i["published"],
#                 "source": i["source"]["href"],
#             }
#         )

#     prompt_scaffold = """
#         Out of the news headlines below, which ones are the most likely corresponding to an article about investments, mergers or acquisitions made by the company {company}? Return the result as a list without any explanations or additional writing.
        
#         Headlines:
        
#         {headlines}
        
#         Example answer:
        
#         ["headline1", "healdline2", "headline3"]
#     """

#     headlines = ""

#     for i in structured:
#         headlines += f"{i['title']}\n\n"

#     prompt_template = PromptTemplate(
#         input_variables=["company", "headlines"],
#         template=prompt_scaffold,
#     )
    
#     # print(headlines)

#     llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

#     chain = LLMChain(llm=llm, prompt=prompt_template)

#     res = chain.run(company=company, headlines=headlines)
    
#     python_list = ast.literal_eval(res)    

#     result = []
    
#     for i in python_list:
#         for j in structured:
#             if i == j["title"]:
#                 result.append(j)
    
#     return result
    
# # if __name__ == "__main__":
# #     print(harvey_helper("ByBox"))